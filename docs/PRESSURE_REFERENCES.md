# Pressure definitions and references

Audited 2026-09-23 against submission revision `3205902f6754ea696a8f66ad58e5032a690c853e`.

Source-backed documentation of pressure definitions and the separately versioned HiLiftAeroML SI export correction. This file is not evaluator configuration, owner approval, or a scoring-support release.

Generated from [pressure-references.json](pressure-references.json). Source confirmations describe evidence, not permission to open submissions.

## Before evaluating

Apply the dataset-prescribed reference to truth and prediction in the same units. Undo training normalization first. Do not fit an offset to test truth or independently mean-center the two fields.

For each complete case: 100 × sqrt(Σ wᵢ (ŷᵢ − yᵢ)² / Σ wᵢ yᵢ²), where y is the dataset-specific evaluation quantity below.

Where required: 100 × Σ wᵢ |ŷᵢ − yᵢ| / Σ wᵢ |yᵢ|. Preserve the metric’s specified spatial weights.

For case-macro relative L1/L2, combine additive chunk sums over the whole case, then average complete-case percentages equally. Keep each metric’s frozen support and weights.

A zero truth norm is undefined and must fail evaluation, including when prediction is also zero. The audited reductions do not add an epsilon or floor a small positive norm; tiny finite denominators remain sensitive. Non-finite inputs/results are invalid. Do not skip, impute or pool a failing case.

Rotor37 benchmark field RRMSE is sqrt(mean_cases(mean_nodes(error²) / max_nodes(|truth|)²)); it is a ratio, not a percentage and not the L2 norm ratio. A zero per-case infinity norm is invalid.

Multiplying truth and prediction by the same nonzero factor leaves within-case relative L1/L2 unchanged. Subtracting a constant changes the denominator. MAE/RMSE retain a unit scale; relative-metric invariance does not validate their units.

## Dataset inventory

| Dataset | Evidence status |
| --- | --- |
| [AhmedML](#ahmedml) | Confirmed from paper and solver setup |
| [AirfRANS](#airfrans) | Confirmed from paper and author library |
| [DrivAerML](#drivaerml) | Confirmed from paper and evaluator |
| [DrivAerNet++](#drivaernetplusplus) | Confirmed by recorded dataset-owner approval |
| [HiLiftAeroML](#hiliftaeroml) | Confirmed; SI dimensional export corrected |
| [WindsorML](#windsorml) | Surface confirmed; volume reference unresolved |
| [Rotor37](#rotor37) | Absolute SI pressure supported by a release sample |
| [VKI-LS59](#vki-ls59) | No directly scored pressure field |

<a id="ahmedml"></a>

## AhmedML

**Confirmed from paper and solver setup.** Surface and volume pressure use kinematic gauge pressure, with the released outlet fixed at zero. Preserve that reference.

### Surface and volume

- **Native field / association:** `pMean`; CellData.
- **Units:** m²/s².
- **Reference:** Outlet fixedValue = 0; inlet velocity = 1 m/s.
- **Current evaluation:** Score the released pMean values without another offset.
- **Conversion:** Cp = pMean / (0.5 × 1²) = 2 × pMean. This scaling leaves relative L1/L2 unchanged.

Use the static-pressure field. total(p)_coeffMean is a different quantity. Absolute-error diagnostics retain their declared native units.

**Sources checked**

- [AhmedML paper](https://arxiv.org/pdf/2407.20801v1), §B.7, Eq. 36; Table 6 (PDF pp. 24, 26): Defines zero reference, density 1, speed 1 m/s, and relative kinematic pMean.
- [Released OpenFOAM case](https://huggingface.co/datasets/neashton/ahmedml/blob/02688c727cdb8dc8678e28abc6bbbb7e93c5fa15/openfoam-casesetup.tgz), 0/p; 0/include/initialConditions: Pressure dimensions [0 2 -2 0 0 0 0]; outlet fixedValue inherits pressure 0.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/ahmedml/submission-spec.json](../benchmark-specs/ahmedml/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [reference/ahmedml/evaluator.py](../reference/ahmedml/evaluator.py): Native pMean enters field accumulators directly; no fitted pressure offset.
- [reference/ahmedml/contract.py](../reference/ahmedml/contract.py): Surface and volume both require native CellData pMean.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="airfrans"></a>

## AirfRANS

**Confirmed from paper and author library.** Surface and 2D flow-domain pressure use kinematic pressure with a zero freestream gauge in the incompressible dataset.

### Airfoil surface and 2D flow domain

- **Native field / association:** `p`; PointData.
- **Units:** m²/s².
- **Reference:** Incompressible freestream pressure = 0. The compressible validation case in the paper uses a different convention.
- **Current evaluation:** Use native p after undoing training standardization; no extra mean subtraction.
- **Conversion:** Cp = p / (0.5 × U∞²), using each case’s inlet speed. Density is already divided out of p.

The paper’s training-set mean/std normalization is a model preprocessing step, not the pressure reference used for field evaluation.

**Sources checked**

- [AirfRANS paper](https://arxiv.org/pdf/2212.07564v3), Appendix J, Tables 7 and 9 (PDF pp. 29, 31): The incompressible field starts at zero with freestreamPressure; absolute atmospheric pressure is only for compressible validation.
- [Official airfrans library](https://github.com/Extrality/airfrans_lib/blob/d35d4035d8ba6fa98c1a6662be925c2fc777610b/src/airfrans/simulation.py), Simulation.__init__; boundary_layer, lines 65–66 and 399–403: Loads native p; the incompressible coefficient divides by 0.5 U∞² without density or another pressure offset.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/airfrans/submission-spec.json](../benchmark-specs/airfrans/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [reference/airfrans_profiles.py](../reference/airfrans_profiles.py): Velocity-profile helper does not define a new pressure gauge.
- [reference/evaluate_predictions.py](../reference/evaluate_predictions.py): Field reductions operate on aligned supplied truth and prediction arrays; they do not infer a gauge.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="drivaerml"></a>

## DrivAerML

**Confirmed from paper and evaluator.** Surface and volume pressure use kinematic pressure on the published zero-outlet gauge. The reference is at (80, 10, 10) m.

### Surface and volume

- **Native field / association:** `pMeanTrim`; CellData.
- **Units:** m²/s².
- **Reference:** p_ref = 0 at (80, 10, 10) m on the outlet.
- **Current evaluation:** Use native pMeanTrim; preserve the outlet reference rather than substituting an upstream or experimental reference.
- **Conversion:** Cp = pMeanTrim / (0.5 × 38.889²); ρ∞ = 1 kg/m³ in this setup.

CpMeanTrim is static pressure coefficient; CptMeanTrim is total pressure coefficient. The uploaded mesh 0/system directories are default ANSA outputs, not the simulation settings.

**Sources checked**

- [DrivAerML paper](https://arxiv.org/pdf/2408.11969v2), §B.2, Eq. 34; §B.3; Tables 3–4: Specifies the outlet reference and warns that moving it changes the pressure values.
- [Pinned dataset README](https://huggingface.co/datasets/neashton/drivaerml/blob/7a5c0948ce27be709b1116a3a190f806e7a8f79f/README.md), File descriptions: openfoam_meshes: Explicitly excludes the supplied default 0/system folders as evidence of the simulation boundary conditions.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/drivaerml/submission-spec.json](../benchmark-specs/drivaerml/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [reference/drivaerml/evaluator.py](../reference/drivaerml/evaluator.py): Uses pMeanTrim directly for surface and volume relative-L2 statistics.
- [reference/drivaerml/accumulators.py](../reference/drivaerml/accumulators.py): Complete-case sums; rejects zero truth norm and non-finite reductions.
- [benchmark-specs/drivaerml/proposal/SCIENTIFIC_CONTRACT.md](../benchmark-specs/drivaerml/proposal/SCIENTIFIC_CONTRACT.md): Documents the native pressure convention and reference speed.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="drivaernetplusplus"></a>

## DrivAerNet++

**Confirmed by recorded dataset-owner approval.** The pressure-only task uses surface kinematic gauge pressure. The owner records a zero freestream reference and Cp = p / 450.

### Surface only

- **Native field / association:** `p`; PointData.
- **Units:** m²/s².
- **Reference:** Zero freestream gauge, recorded in the dataset-owner approval.
- **Current evaluation:** Use native p on every required surface point; undo any training mean/std scaling.
- **Conversion:** Cp = p / 450 at U∞ = 30 m/s. Gauge pressure in Pa = 1.184 × p.

The author README calls the surface modality Cp, but its loader reads p directly. Use the explicit owner convention, not the modality label. This audit does not activate wall-shear or volume tasks.

**Sources checked**

- [DrivAerNet++ paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/013cf29a9e68e4411d0593040a8a1eb3-Paper-Datasets_and_Benchmarks_Track.pdf), Appendix B; Table 6 (PDF pp. 18–20): Documents simpleFoam and the 30 m/s, 1.184 kg/m³ flow conditions. It does not alone establish the exact archived p gauge.
- [Author surface-pressure loader](https://github.com/Mohamedelrefaie/DrivAerNet/blob/8b8c053053aa0f27c30f73af00be070a20205858/RegDGCNN_SurfaceFields/data_loader.py), sample_point_cloud_with_pressure, lines 66–87: Reads PointData p without coefficient conversion. Its sampled training loader does not define FluidsBench scoring support.
- [Recorded owner approval](https://github.com/neilashton/fluidsbench-submission/blob/cb64dacb296e467bc706de19093142c66b4056da/benchmark-specs/drivaernetplusplus/OWNER_APPROVAL.md), What is approved; 4 September 2026: Records kinematic gauge pressure, zero freestream reference, and Cp = p / 450.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/drivaernetplusplus/submission-spec.json](../benchmark-specs/drivaernetplusplus/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [benchmark-specs/drivaernetplusplus/OWNER_APPROVAL.md](../benchmark-specs/drivaernetplusplus/OWNER_APPROVAL.md): Existing scientific approval; final support publication remains separate.
- [reference/evaluate_predictions.py](../reference/evaluate_predictions.py): Pressure arrays are reduced without a fitted offset.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="hiliftaeroml"></a>

## HiLiftAeroML

**Confirmed; SI dimensional export corrected.** Relative surface and volume pressure errors use Cp = (P − p∞) / q∞. Raw pressure is in the solver’s slug–inch–second units.

### Surface

- **Native field / association:** `PROJ(AVG(P))`; PointData.
- **Units:** Native slug/(in·s²); Cp is dimensionless.
- **Reference:** Use the same per-case p∞ and q∞ = 0.5 ρ∞ |U∞|² as the frozen evaluator.
- **Current evaluation:** Relative L1/L2 use Cp after undoing model standardization.
- **Conversion:** Subtract p∞ and divide by q∞ in one consistent unit system.

### Volume

- **Native field / association:** `avg(P)`; PointData.
- **Units:** Native slug/(in·s²); Cp is dimensionless.
- **Reference:** Same per-case freestream reference as the surface.
- **Current evaluation:** Apply the published raw Float32 avg(P) != 0 validity mask before normalization; relative L1/L2 use Cp.
- **Conversion:** The package export converts native pressure and surface wall-shear MAE/RMSE by × 574.5631077637795 into Pa, and native velocity MAE/RMSE by × 0.0254 into m/s. Each case keeps its original inverse normalization before equal-case averaging. Apply this SI export exactly once.

The native profile materializer computes Cp from raw pressure and per-case freestream metadata. The per-case ref_values CSV supplies qRef for load normalization; do not substitute it for the field evaluator’s training q∞.

The current ICLR manuscript explicitly states the SI conversions. Its bundled archive exposed a native-unit mismatch in the original FluidsBench dimensional diagnostics. The versioned hiliftaeroml-dimensional-export-si-v1 correction now fixes the assembler and all 23 existing preview records. Original validation receipts are preserved.

All 184 corrected aggregate MAE/RMSE values match the manuscript across 8,142 case records. Full-split GeoTransolver surface-pressure RMSE is 87.3107201811254 Pa and volume-velocity RMSE is 2.6561332906973014 m/s.

These checks replay exported sufficient statistics and dimensional metrics; they do not rerun model inference. Relative errors, R², coefficients, overall scores, ordering, profiles and nondimensional regional diagnostics are unchanged. The pressure reference, weights and validity masks are unchanged.

The catalog source snapshot is bbec30b; the field-support archive identity retains 1c266d3. The SI export correction does not repin either support release or open submissions.

**Sources checked**

- [HiLiftAeroML paper](https://arxiv.org/pdf/2605.19565v1), Appendix B.1, Eqs. 9–18; Table 11: Defines English solver units and the freestream reference; distinguishes surface PROJ(AVG(P)) and volume avg(P).
- [Pinned per-case references](https://huggingface.co/datasets/nvidia/HiLiftAeroML/blob/bbec30bcfc6103309c1375c5228b3ad0a586bfaf/geo_LHC001_AoA_4/ref_values_geo_LHC001_AoA_4.csv), geo_LHC001_AoA_4: qRef = 4.937856000000001 and uRef = 2679.5054741899685 in native units. The CSV bytes also match the field-support source revision 1c266d3869bc2968ff97d2107c9c3919be03ed32.
- [Author PhysicsNeMo recipe](https://github.com/NVIDIA/physicsnemo/blob/426f7552da4b4fa675e404e8a4f437e27681b668/examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets/highlift_volume.yaml), Dataset header, lines 17–25: States that per-case freestream metadata uses slug-inch-second-Rankine. This is corroborating author code, not a replacement for the frozen evaluator.
- [NIST SI conversion factors](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8), Entries for slug and inch: 1 slug ≈ 14.59390 kg and 1 in = 0.0254 m. Their ratio gives ≈ 574.563 Pa per slug/(in·s²); this dimensional conversion is derived here, not evidence of its implementation in the evaluator.
- [Current ICLR manuscript and reproducible unit audit (author project; access required)](https://www.overleaf.com/project/6aa9a4e94510283d02e65151), Overleaf source inspected 2026-09-23; sections/appendix-ml.tex, Physical variables and Absolute errors in engineering units; scripts/build_hilift_absolute_error_assets.py; tables/ml_native_absolute_errors_audit.json: Explicit SI conversions and a hash-checked archive reproduce 23 records and 8,142 case records. The manuscript corrects native-unit exports while preserving per-case normalization. The inspected appendix and script match the local manuscript checkout byte for byte.
- [Original FluidsBench result export](https://github.com/neilashton/fluidsbench-submission/blob/cb64dacb296e467bc706de19093142c66b4056da/leaderboard/all.json), 23 HiLiftAeroML rows and their linked case-metric files; compared with the paper’s frozen e19809a archive: Before correction, all HiLiftAeroML metric values matched the archived export and all 23 case-file hashes matched. Recovered dimensional scales of about 4.93823 for pressure/shear and 2679.505 for velocity confirmed native units despite SI labels.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/hiliftaeroml/submission-spec.json](../benchmark-specs/hiliftaeroml/submission-spec.json): Keeps Cp for relative pressure errors and now states the explicit native-to-SI dimensional export factors.
- [reference/hiliftaeroml/native_profile_truth_materializer.py](../reference/hiliftaeroml/native_profile_truth_materializer.py): Computes q∞ from per-case metadata and Cp from raw pressure.
- [benchmark-specs/hiliftaeroml/evaluator-implementation-manifest.candidate.json](../benchmark-specs/hiliftaeroml/evaluator-implementation-manifest.candidate.json): Binds external inference/metric implementation hashes; source roots still have pending candidate revision bindings.
- [scripts/prepare_hiliftaeroml_schema_v3_candidate.py](../scripts/prepare_hiliftaeroml_schema_v3_candidate.py): Generates the explicit per-case inverse-normalization and native-to-SI export convention.
- [benchmark-specs/hiliftaeroml/dimensional-export-si-v1.json](../benchmark-specs/hiliftaeroml/dimensional-export-si-v1.json): Versioned eight-metric SI export contract; relative statistics and scoring are unchanged.
- [benchmark-specs/hiliftaeroml/dimensional-export-correction-v1.json](../benchmark-specs/hiliftaeroml/dimensional-export-correction-v1.json): Pins original and corrected metadata, evidence, case files and deterministic archives for 23 preview packages.
- [reference/hiliftaeroml/dimensional_units.py](../reference/hiliftaeroml/dimensional_units.py): Converts only dimensional field errors, rejects repeated conversion and checks the immutable correction registry.
- [scripts/assemble_hiliftaeroml_schema_v3_candidate.py](../scripts/assemble_hiliftaeroml_schema_v3_candidate.py): Applies the SI conversion to each completed native case before macro aggregation and records the separate export version.
- [scripts/validate_submission.py](../scripts/validate_submission.py): Requires the SI export evidence and retains the original closed-candidate validation receipts when checking corrected packages.
- [benchmark-specs/hiliftaeroml/dimensional-export-correction-v1/verification.json](../benchmark-specs/hiliftaeroml/dimensional-export-correction-v1/verification.json): Records the independent manuscript comparison and unchanged scores, ordering, case statistics and non-dimensional artifacts; states test limits.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="windsorml"></a>

## WindsorML

**Surface confirmed; volume reference unresolved.** Surface cpavg is a pressure coefficient. The volume pressureavg reference and units need clarification before any pressure subtraction is prescribed.

### Surface

- **Native field / association:** `cpavg`; PointData.
- **Units:** Dimensionless.
- **Reference:** Published nominal reference: p_ref = 37330.4 Pa, ρ_ref = 1.31 kg/m³, U_ref = 40 m/s.
- **Current evaluation:** Score released cpavg directly. The reference has already been applied.
- **Conversion:** Cp = (P − 37330.4) / 1048 for dimensional P in Pa under the published nominal convention.

### Volume

- **Native field / association:** `pressureavg`; CellData.
- **Units:** Unresolved: paper labels m²/s²; sampled export needs reconciliation.
- **Reference:** No additional reference subtraction is implemented by the current evaluator. The tunnel exit is not a fixed-pressure boundary.
- **Current evaluation:** The current evaluator compares raw pressureavg values. Preserve that fact without treating it as a confirmed physical convention.
- **Conversion:** Do not apply the surface Cp conversion to pressureavg until the export units and offset are confirmed.

For the dataset, the paper uses nominal inlet conditions; the numerical probe at (−2, 1.3, 0) m is used for the separate experimental validation.

Table 6 calls pressureavg relative kinematic pressure. The pinned run_1 volume XML reports a range of 33105.16015625 to 38447.64453125. Those magnitudes suggest an absolute dimensional export, but a range alone cannot establish units or gauge. The solver/export implementation is not published with this release.

**Unresolved evidence / required closure**

- Dataset owner/Volcano: confirm the exact pressureavg export expression (P, P/ρ, (P−p_ref)/ρ, or another quantity), density convention, units and reference. Reconcile it with surface cpavg on a small paired sample. If scoring should use a different offset, replay affected metrics and issue a new evaluator/result release.

**Sources checked**

- [WindsorML paper](https://arxiv.org/pdf/2407.19320v4), §B.6.4–B.6.5; Tables 5–6 (PDF pp. 19–20): Defines surface Cp and nominal references, distinguishes the validation probe, and labels volume pressureavg relative kinematic pressure.
- [Pinned WindsorML dataset](https://huggingface.co/datasets/neashton/windsorml/blob/8a6ca32ae22c94f54df2186d1b0ccf9662a294c2/README.md), CFD Solver and files: Identifies the commercial Volcano solver and native volume/boundary files.
- [Pinned native volume](https://huggingface.co/datasets/neashton/windsorml/blob/8a6ca32ae22c94f54df2186d1b0ccf9662a294c2/run_1/volume_1.vtu), CellData pressureavg XML RangeMin/RangeMax: Read with bounded HTTP ranges, skipping compressed payloads. The header range is evidence of an ambiguity, not a measured whole-file unit conversion.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/windsorml/submission-spec.json](../benchmark-specs/windsorml/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [reference/windsorml/contract.py](../reference/windsorml/contract.py): Surface cpavg and volume pressureavg are separate native fields.
- [reference/windsorml/evaluator.py](../reference/windsorml/evaluator.py): Both scalar arrays enter statistics directly; no volume pressure-reference subtraction.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="rotor37"></a>

## Rotor37

**Absolute SI pressure supported by a release sample.** Blade-surface Pressure is consistent with absolute static pressure in Pa. Preserve the released pressure; do not subtract a freestream or case mean.

### Blade surface

- **Native field / association:** `Pressure`; CGNS GridLocation Vertex.
- **Units:** Pa, supported by sampled equation-of-state consistency; explicit owner declaration pending.
- **Reference:** Absolute thermodynamic pressure is supported by P/(ρT) ≈ 287 across one public sample.
- **Current evaluation:** Use native Pressure for relative L1/L2 and the separately defined benchmark RRMSE. Do not turn it into a gauge coefficient.
- **Conversion:** No pressure offset or Cp conversion. The scalar input P is not an instruction to subtract that value from the output field.

Inspected public row 100: 29,773 vertices; P ranges from 9889.54 to 238060.26, and P/(Density × Temperature) is 287 within floating-point precision. The inspected CGNS tree has no explicit units declaration.

The papers and dataset card describe compressible blade-surface pressure, density and temperature. They do not fully specify the pressure boundary input or its dimensional convention. One sample is supporting evidence, not a release-wide verification.

**Unresolved evidence / required closure**

- Safran dataset owner: explicitly record Pressure as absolute static pressure, its units, and the boundary meaning/units of input P. Keep the current raw-field evaluation unless an approved correction establishes otherwise.

**Sources checked**

- [MMGP paper](https://arxiv.org/pdf/2305.12871v2), §4.1 and Appendix A.1: Introduces the Rotor37 blade-surface pressure task; no explicit pressure unit/reference definition was found.
- [PLAID datasets paper](https://arxiv.org/pdf/2505.02974v3), §4.3.1: Describes compressible elsA calculations and output Density, Pressure and Temperature.
- [Pinned Rotor37 release](https://huggingface.co/datasets/PLAID-datasets/Rotor37/blob/bac06c0caa7254120eecc6711a5fb85c58dfbdbc/data/all_samples-00000-of-00011.parquet), Public row 100, native Vertex arrays: Static inspection of serialized numeric buffers gave P = 287 ρT. The record does not add a pressure offset.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/rotor37/submission-spec.json](../benchmark-specs/rotor37/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [benchmark-specs/rotor37/methodology-contract.json](../benchmark-specs/rotor37/methodology-contract.json): Maps Pressure to blade-surface metrics.
- [reference/evaluate_predictions.py](../reference/evaluate_predictions.py): RRMSE uses per-case maximum absolute truth for normalization; it is not relative L2.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

<a id="vki-ls59"></a>

## VKI-LS59

**No directly scored pressure field.** The current task scores Mach number, turbulent viscosity, blade isentropic Mach number and scalar outputs. It has no direct pressure-field metric.

### 2D flow domain and blade curve

- **Native field / association:** `mach, nut; M_iso on the blade curve`; CGNS GridLocation Vertex.
- **Units:** Mach quantities are dimensionless; pressure is not a direct target.
- **Reference:** No pressure gauge is applied to the current scored fields.
- **Current evaluation:** Use the released fields and the separate scalar Pr contract. Pr is a pressure-ratio output, not a local pressure field.
- **Conversion:** Do not reconstruct and add a pressure target from ro, rou, rov and roe as part of this audit.

The release also contains conservative flow variables. Defining a future derived pressure field would require the owner’s gas model, nondimensionalization, reference state and M_iso/Pr extraction conventions.

**Sources checked**

- [PLAID datasets paper](https://arxiv.org/pdf/2505.02974v3), §4.3.3 and Appendix C.2: Describes compressible BROADCAST data and the separate flow-domain/blade-curve supports.
- [Pinned VKI-LS59 dataset card](https://huggingface.co/datasets/PLAID-datasets/VKI-LS59/blob/1aad0a69c26462c039305a931a300f80bdf34827/README.md), Problem definition and sample access: Lists mach, nut, M_iso and scalar outputs; no direct pressure-field target.
- [Author dataset release](https://zenodo.org/records/14840512), Dataset description and access example: Provides conservative fields and M_iso on a distinct blade curve.

<details>
<summary>Evaluator and specification trace</summary>

- [benchmark-specs/vki-ls59/submission-spec.json](../benchmark-specs/vki-ls59/submission-spec.json): The audited dev scoring specification; not modified by this documentation audit.
- [benchmark-specs/vki-ls59/methodology-contract.json](../benchmark-specs/vki-ls59/methodology-contract.json): Current scored fields contain no direct pressure field.
- [reference/evaluate_predictions.py](../reference/evaluate_predictions.py): No pressure reconstruction or pressure-gauge fitting is performed.

SHA-256 identities for these files and downloaded sources are recorded in the JSON inventory.

</details>

## Audit limits and next steps

The WindsorML header inspection, Rotor37 sample calculation and HiLiftAeroML exported-metric replay are recorded in [pressure-reference-observations.json](pressure-reference-observations.json). These checks do not rerun full-dataset native inference.

Resolve WindsorML’s volume export definition and obtain Rotor37’s explicit owner declaration. HiLiftAeroML’s pressure convention is confirmed and its dimensional export is corrected through `hiliftaeroml-dimensional-export-si-v1`; see [the correction record](../benchmark-specs/hiliftaeroml/DIMENSIONAL_EXPORT_CORRECTION.md). No author messages were sent by this audit.

A changed offset or unit conversion needs source evidence, an explicit evaluator or export version, recomputation of affected metrics and a distinct result release. Preserve original artifacts and receipts. The HiLiftAeroML correction follows this process without changing the pressure offset, weights, split, validity mask, threshold or score. The other dataset findings remain documentation only.

To check the audit inputs and regenerate this document:

```bash
python3 scripts/render_pressure_reference_docs.py --check
```

Omit `--check` to render. Add `--website-root /path/to/fluidsbench` to synchronize or check the website’s exact JSON copy.
