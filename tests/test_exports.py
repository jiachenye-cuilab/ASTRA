import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from astra.inference.export_diagnostics import fov_grid_diagnostics, section_grid_diagnostics, spot_grid_weights
from astra.inference.section import predict_section, open_fields
from astra.model.segment import build_segment_layout
from astra.model.fp32 import allocate_fp32
from astra.model.allocation import aggregate_segments
from astra.inference.section_geometry import complete_spot_owner
from fixtures import synthetic_package, synthetic_section


class GridTests(unittest.TestCase):
    def fov(self, owner, valid, grid, counts, parent_valid):
        genes = np.array(['synthetic_A'])
        data = dict(gene_ids=genes, gene_available=np.ones(1,bool), owner_map=owner,
                    field_valid=valid, parent_counts=np.asarray(counts).reshape(-1,1), parent_valid=np.array(parent_valid,bool))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'export.npz'
            np.savez(path, pred_count_8um=np.asarray(grid,np.float32).reshape(owner.shape[0]//4,owner.shape[1]//4,1),
                     gene_ids=genes, gene_available=data['gene_available'])
            return fov_grid_diagnostics(path, data)

    def test_nonuniform_aligned_grid(self):
        owner = np.repeat(np.repeat(np.array([[0,0],[1,1]]),4,axis=0),4,axis=1)
        result = self.fov(owner, np.ones_like(owner,bool), [1,9,2,18], [10,20], [True,True])
        self.assertTrue(result['exact_representation'])
        self.assertEqual(result['max_absolute'], 0.)

    def test_partial_valid_cell_uses_valid_area_denominator(self):
        owner = np.zeros((4,4),np.int64)
        valid = np.zeros((4,4),bool); valid[:2,:2] = True
        result = self.fov(owner,valid,[10],[10],[True])
        self.assertEqual(result['max_absolute'], 0.)

    def test_empty_support_is_not_a_pass_claim(self):
        result = self.fov(np.full((4,4),-1),np.zeros((4,4),bool),[0],[0],[False])
        self.assertEqual(result['status'], 'not_evaluated')
        self.assertEqual(result['compared_values'], 0)

    def test_exact_allocation_does_not_imply_exact_grid(self):
        owner = torch.full((1,4,4),-1,dtype=torch.long); owner[0,0,0] = 0
        valid = torch.ones_like(owner,dtype=torch.bool)
        layout = build_segment_layout(owner,valid,parent_slots=1)
        mass, total, residual = allocate_fp32(torch.tensor([[15.],[1.]]),torch.zeros(2,1),layout,
            torch.tensor([[[10.]]]),torch.tensor([[True]]),tolerance=5e-6,validate=True)
        grid = aggregate_segments(mass,layout,1).numpy()
        self.assertEqual(residual.item(), 0.)
        self.assertEqual(total.item(), 10.)
        result = self.fov(owner[0].numpy(), valid[0].numpy(), grid, [10], [True])
        self.assertFalse(result['exact_representation'])
        self.assertFalse(result['within_tolerance'])
        self.assertEqual(result['max_absolute'], 8.4375)

    def test_circle_operator_matches_independent_2um_raster(self):
        center = np.array([55.3,61.7])
        cells, weights = spot_grid_weights(center)
        yy, xx = np.indices((64,64))
        mask = (2*yy+1-center[0])**2 + (2*xx+1-center[1])**2 <= 27.5**2
        independent = mask.reshape(16,4,16,4).sum((1,3))/16
        np.testing.assert_array_equal(weights, independent[cells[:,0],cells[:,1]])
        # Constant density is a sufficient exactness condition even across circle boundaries.
        self.assertAlmostEqual(weights @ np.full(len(weights),64*0.75), mask.sum()*4*0.75)

    def test_overlapping_measured_spots_are_rejected(self):
        with self.assertRaises(ValueError):
            complete_spot_owner([[72,72],[80,80]],fov_origin_yx_um=[0,0])


class SectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_section(self, task, *, core=160, partial=False):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        root = Path(d.name)
        package, genes, metadata = synthetic_package(root/'package')
        cache = synthetic_section(root/'cache',genes,metadata,task,core=core,partial=partial)
        output = root/'output'
        report = predict_section(cache,output,package,device='cpu',batch_size=2,pipeline='cpu')
        # Reopen the final report and arrays, independently of the live memmap.
        self.assertEqual(json.loads((output/'report.json').read_text())['serialized_export'], report['serialized_export'])
        return output, open_fields(cache,genes,metadata,'cpu'), report

    def test_hd16_end_to_end_boundary_and_nonuniform_counts(self):
        output, fields, report = self.run_section('HD16')
        self.assertTrue(report['serialized_export']['within_tolerance'])
        self.assertLess(report['serialized_export']['max_scaled'], 1e-5)
        grid = np.load(output/'prediction.npy')
        self.assertGreater(np.ptp(grid), 0.)
        grid[0,0,0] += 1
        np.save(output/'prediction.npy',grid)
        with self.assertRaisesRegex(ValueError,'does not conserve'):
            section_grid_diagnostics(output,fields)

    def test_spot55_overlapping_export_reports_raster_residual(self):
        output, fields, report = self.run_section('Spot55')
        diagnostic = report['serialized_export']
        self.assertFalse(diagnostic['exact_representation'])
        self.assertEqual((diagnostic['complete_spots'],diagnostic['partial_spots_excluded'],diagnostic['empty_spots_excluded']), (2,1,1))
        self.assertIsNone(report['export_conservation_tolerance'])
        self.assertGreater(diagnostic['max_absolute'], 1e-5)
        # Independent whole-raster integral of the serialized grid, not allocation tensors.
        grid = np.load(output/'prediction.npy'); query = np.load(output/'query_yx_8um.npy')
        raster = np.zeros((32,32,2)); raster[query[:,0],query[:,1]] = grid
        fine_density = raster.repeat(4,0).repeat(4,1)/16
        yy,xx = np.indices((128,128))
        sums = []
        for y,x in [[72,72],[144,144]]:
            mask = (2*yy+1-y)**2+(2*xx+1-x)**2 <= 27.5**2
            sums.append(fine_density[mask].sum(0))
        expected = np.max(np.abs(np.array(sums)-np.array([[19,37],[53,71]])))
        self.assertAlmostEqual(diagnostic['max_absolute'],expected,places=10)

    def test_spot55_nonoverlapping_export_is_also_approximate(self):
        _, _, report = self.run_section('Spot55',core=144)
        self.assertEqual(report['serialized_export']['complete_spots'],2)
        self.assertFalse(report['serialized_export']['exact_representation'])

    def test_partial_queries_do_not_compare_with_full_spot_counts(self):
        _, _, report = self.run_section('Spot55',partial=True)
        diagnostic = report['serialized_export']
        self.assertEqual(diagnostic['complete_spots'],1)
        self.assertGreater(diagnostic['partial_spots_excluded'],0)


if __name__ == '__main__':
    unittest.main()
