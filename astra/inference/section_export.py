import numpy as np
import torch

class ExportPlan:
    def __init__(self, task, fields, indices, batch_size=4):
        self.task, self.fields, self.indices = task, fields, indices
        self.assignment = np.load(fields.root / ('assignment.npy' if task == 'HD16' else 'query_assignment.npy'))
        order = np.argsort(self.assignment, kind='stable')
        ends = np.r_[0, np.cumsum(np.bincount(self.assignment, minlength=fields.fields))]
        self.slots = {f: order[ends[f]:ends[f + 1]] for f in indices}
        if task == 'HD16':
            self.local = np.load(fields.root / 'local_parent.npy')
            destinations = {f: self.slots[f] for f in indices}
        else:
            self.cells = np.load(fields.root / 'query_yx_8um.npy')
            self.query = np.load(fields.root / 'query_index.npy')
            weights = np.load(fields.root / 'query_weight.npy')
            self.weights = weights.astype(np.float32)
            np.testing.assert_array_equal(self.weights, weights)
            destinations = {f: self.query[self.slots[f]] for f in indices}
        self.unique = np.unique(np.concatenate(list(destinations.values())))
        self.destination = {f: np.searchsorted(self.unique, destinations[f]) for f in indices}
        self.expected_weight = np.zeros(len(self.unique))
        if task == 'Spot55':
            for f in indices:
                self.expected_weight[self.destination[f]] += self.weights[self.slots[f]]
        self.batches = {}
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            packed, offsets = [], [0]
            for i, f in enumerate(selected):
                slots = self.slots[f]
                if task == 'HD16':
                    parent = self.local[slots]
                    y = (parent // 16)[:, None] * 2 + np.array([0, 0, 1, 1])
                    x = (parent % 16)[:, None] * 2 + np.array([0, 1, 0, 1])
                    local = (y * 32 + x).ravel()
                else:
                    yx = self.cells[self.query[slots]] - fields.starts[f] // 4
                    local = yx[:, 0] * 32 + yx[:, 1]
                packed.append(i * 1024 + local)
                offsets.append(offsets[-1] + len(local))
            self.batches[tuple(selected)] = (
                torch.as_tensor(np.concatenate(packed), device=fields.device), offsets)

    def shape(self, genes):
        return (len(self.unique), 4, genes) if self.task == 'HD16' else (len(self.unique), genes)

    def compact(self, result, inputs, packet, model, selected, prediction, accumulated):
        pred = result['pred_count_8um']
        take, offsets = self.batches[tuple(selected)]
        values_gpu = pred.reshape(-1, pred.shape[-1]).index_select(0, take)
        if self.task == 'HD16':
            # Match NumPy's FP32 TL/TR/BL/BR sequential sum, including non-exported parents.
            p = pred.float()
            total = ((p[:, 0::2, 0::2] + p[:, 0::2, 1::2]) + p[:, 1::2, 0::2]) + p[:, 1::2, 1::2]
            truth = model.select_output_genes(packet.parent_counts)
            known = inputs['parent_valid']
            relative = (total.flatten(1, 2).float() - truth.float()).abs() / (1 + truth).float()
            error = relative.masked_fill(~known[..., None], 0).max()
            # Every exported child must belong to a complete observed parent.
            parent_support = known.reshape(-1, 16, 16).repeat_interleave(2, 1).repeat_interleave(2, 2)
            support_ok = parent_support.flatten().index_select(0, take).all()
            valid_values = torch.isfinite(p).all() & (p >= 0).all()
        else:
            residual = result['parent_conservation_residual'].abs() / (1 + model.select_output_genes(packet.parent_counts).abs())
            error = residual.max()
            support = packet.field_valid.reshape(len(selected), 32, 4, 32, 4).all((2, 4))
            support_ok = support.flatten().index_select(0, take).all()
            valid_values = torch.isfinite(values_gpu).all() & (values_gpu >= 0).all()
        diagnostics = torch.stack((error.float(), support_ok.float(), valid_values.float())).cpu().numpy()
        assert diagnostics[0] <= 1e-5 and diagnostics[1] == diagnostics[2] == 1, diagnostics
        # One packed prediction copy per batch; no early FP32 cast for Spot55.
        values = values_gpu.cpu().numpy()
        for i, f in enumerate(selected):
            part = values[offsets[i]:offsets[i + 1]]
            if self.task == 'HD16':
                prediction[self.destination[f]] = part.reshape(-1, 4, pred.shape[-1]).astype(np.float32)
            else:
                slots = self.slots[f]
                prediction[self.destination[f]] += part * self.weights[slots, None]
                accumulated[self.destination[f]] += self.weights[slots]
        return float(diagnostics[0])
