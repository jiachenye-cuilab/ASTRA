# Conservation, support and serialized grids

Counts are inferred **UMI mass per spatial support and gene**, not concentration
or measured RNA at 8 µm. Coordinates are `[y,x]`; transforms act on `[col,row,1]`.
A native 2 µm pixel `(i,j)` has physical edges `[2i,2i+2] × [2j,2j+2]` µm and
center `(2i+1,2j+1)` µm. An 8 µm cell contains 4×4 such pixels (64 µm²).
The calibrated `spot_to_image` maps native pixel-center indices to source image
pixel-center coordinates. Dimensions and generic DPI do not define physical scale.

## Observation and allocation operators

Let `V_f` be valid capture/image support in FOV `f`, `S_fk` an observation's
support, and `Q_fq` an 8 µm cell. The model partitions `S_fk ∩ Q_fq ∩ V_f`
into segments. Segment area is **4 µm² times the number of valid 2 µm pixels**;
this is center rasterization, not analytic continuous circle/pixel overlap.
Gap segments (`owner=-1`) have inferred mass without count labels.

- **HD16:** aligned 16 µm squares, 8×8 native pixels, four complete 8 µm children.
  Only completely captured, registered, observed parents export. Missing parents
  and genes are masks, not measured zeros.
- **Spot55:** continuous centers, diameter 55 µm, radius 27.5 µm. A circle must
  fit entirely in its 256 µm FOV before rasterization. Pixels with centers
  inside/on the circle are included. A retained circle needs complete valid
  raster support; partial circles are dropped without cropping their counts.
  Intersecting measured spot masks are rejected by the owner-map builder.
  Continuous acquisition counts are assigned to this discrete support, whose
  area need not equal `π × 27.5²` µm². This geometric approximation is distinct
  from the final-grid mass residual.

For valid parents and measured genes, the allocator guarantees
`sum_q m_fkq = y_fk` within FP32 tolerance. `m_fkq` is the observation/cell
**intersection mass**. The regular-grid output is
`z_fq = sum_k m_fkq + m_f,gap,q`, discarding the within-cell owner partition.

The regular-grid uniform-density observation operator is
`B_fkq = area(S_fk ∩ Q_fq ∩ V_f) / area(Q_fq ∩ V_f)`, zero when the denominator
is zero. Whole-section queries are complete cells: denominator **64 µm²**.
FOV diagnostics use the input's 2 µm `field_valid`, including partially valid
cells. The serialized `field_valid_8um` is the stricter *all 16 pixels valid*
analysis mask and cannot substitute for the original support in that case.

Generally **`B_f z_f != y_f`**: a cell can contain spot/gap/other-owner masses at
different inferred densities. Area weighting cannot recover those lost masses.
An exact internal residual of zero therefore does not prove Spot55 grid
conservation. The deterministic counterexample allocates 10 UMI to one pixel
and 15 UMI to the other 15 pixels: `z=25`, `Bz=25/16=1.5625`, residual
`-8.4375` UMI.

## Stitching and valid guarantees

HD16 assigns one core per complete parent, exports TL/TR/BL/BR children without
averaging, and preserves the original observation operator when those four
serialized children are summed.

Spot55 exports only requested queries, without filling unrequested gaps.
For a query, `z_q = sum_f w_fq z_fq`, where
`w_fq = 1 / number_of_contributing_cores(q)` and `sum_f w_fq = 1`.
These are stitching weights, not observation overlap areas. Default 160/144 µm
cores/strides average overlaps; 144/144 uses one core per query. Both can split
a spot between FOV allocations. Neither restores the lost intersection
partition; spatially varying averaging weights can additionally change totals.
Negative padded context is masked to capture/image support. No clipped support
is compared to a full measured count.

Sufficient exactness conditions include complete aligned HD16 children, or an
unchanged uniform-density map with counts defined on the identical raster
support. Transferring per-FOV equality through averaging also requires the same
full observation/operator and weights constant over that support, summing to
one. Tests exercise sufficient conditions; they are not assumed for Spot55.
No normalization, redistribution or prediction correction is performed.

## Serialized diagnostics and tolerances

`parent_conservation_max_scaled` keeps its historical **pre-export** meaning:
`max(abs(error)/(1+abs(count)))` on allocation masses. Its scope is explicit.
Model allocation checks use `5e-6`; packed HD16 checks use `1e-5`.
After serialization, `serialized_export` reopens NPZ/NPY files:

- FOVs report `Bz-y` on the input support and measured panel. Homogeneous cells
  retain owner identity and must meet the exactness tolerance.
- HD16 sections compare four serialized children with the matching sparse
  coarse row; shape, coordinates, gene order and availability are checked.
- Spot55 sections build the same center-rasterized circles from original
  observations and integrate the final grid with `B`. Only **entire discrete
  supports** present in the exported queries are compared. Partial/empty
  supports are counted separately, with coverage fractions. No complete support,
  or a legacy external cache without original observations, gives
  `status=not_evaluated`. Large residuals are retained, without an exactness gate.

Metrics include absolute UMI error, scaled error, relative error for nonzero
counts and the number compared. Zero-count errors remain in absolute/scaled
metrics. Exact checks use **`atol=1e-5` UMI, `rtol=1e-5`**:
`abs(error) <= atol + rtol*abs(y)`, retaining `1e-5*(1+abs(y))`.
The allocation budget `5e-6` plus three FP32 HD16 additions (unit roundoff
`2^-24`, less than `3.6e-7` relative) fits inside this threshold. Serialization
does not quantize FP32 further; diagnostic accumulation uses FP64. This is a
roundoff tolerance, not a bound on Spot55 representation or biological error.

`tests/test_exports.py` covers serialized readback, nonuniform values, aligned
exactness, partial/empty support, boundaries, overlapping/non-overlapping cores,
the counterexample and an independent whole-raster integral. Synthetic sections
use deterministic random weights and synthetic features, not UNI/RNA ground
truth. Bundled pretrained replay is separate and explicitly enabled.
