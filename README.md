# SSTER synthetic data generator

This package implements the November 25 direct-state profile for jointly supervised anomaly detection and multi-root localization. Production generates 6,720,000 training windows and 336,000 windows in each held-out split. The published input contains direct observations, an acquisition mask, physical time and a valid-channel mask. Fault records, paired healthy states and proof bounds are private.

## Generate

Use Python 3.10 or later. For deterministic reproduction, record environment versions, backend and CUDA hardware along with the calibration checksum. The production extra enables the compiled lagged state solver and optional batched GPU GP sampling. Set the BLAS environment before launch.

~~~bash
python -m pip install -e '.[production,test]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SSTER_GP_DEVICE=cuda
python -m sster_data generate --out /data/sster --split all --workers 6
python -m sster_data verify --out /data/sster --split all --deep
python -m sster_data diagnose --out /data/sster --split train
~~~

Omit the GPU setting for CPU GP evaluation. Workers are spawned, and CUDA GPs use the current worker's selected device. Repeat the generation command after an interruption. Use a separate output directory for the smoke corpus. This exercises every structural family, healthy and joint fault paths, and up to six roots, using 48 windows per split. Smoke data has a separate calibration bank and cannot complete a production release.

~~~bash
python -m sster_data smoke --out /tmp/sster-smoke --split all --workers 2
python -m sster_data verify --out /tmp/sster-smoke --split all --deep
python -m pytest
~~~

## Corpus and random access

Each split has a manifest, an audit report and a completion marker named `_SUCCESS`. Each manifest row gives a relative tar path and byte offset/size pairs `input_offset,input_size` and `target_offset,target_size`. Seek to each offset, read the indicated byte count and load those bytes with NumPy and `allow_pickle=False`. These offsets refer to member data in an uncompressed tar. Arrays within each member use ZIP compression. Each unpadded example has its own length and width.

| Member | Arrays |
|---|---|
| `<id>.inputs.npz` | float32 values [T,N], bool observed [T,N], float64 time [T], bool channel_valid [N] |
| `<id>.targets.npz` | bool Y and Y_known [T], bool R, R_known and E [N] |

For loss computation, apply time supervision on Y_known and channel supervision on R_known intersect E. Use the four input arrays as the model's input. Metadata such as the anchor, coverage ticket, fault count, ancestor identity, and replay fields is restricted to data loading, auditing or synthetic evaluation. A physical world is identified by (N, family, ancestor). Training, validation and test use disjoint hash buckets, and the frozen normal calibration bank has its own namespace.

Each `private/*.tar` carries the physical event ledger, export-axis bijection, realization streams/retry numbers, clipping and normalization parameters, and a per-example schema index. An evenly spaced audit subset also includes full physical float64 normal/factual trajectories, bounds, direct ledger and point labels. Use the replay record and the published readout permutation to regenerate any other example. The runtime baseline priors, event templates and quotas are in the packaged profile. `audit.json` reports achieved template quotas, low witness counts and both channel exposure definitions.

The `diagnose` command uses 64 coupled dense/sparse realizations per kernel by default, reports ACF at lags 1, 4 and 16 and three frequency-band power shares for the raw GP and the final healthy state, and flags differences above 0.10. On the fixed audit episodes it also gives normal-prior frequencies, their conditional distributions at seasonal injection sites, baseline state and innovation statistics, and 1,000 ancestor-cluster bootstrap intervals. The release audit includes per-template event-specific witness counts, severity and layout counts, late re-opening of propagation, right censoring, thinning and missing-witness counts. The following command checks all nine arrays against their original decoded values, using the same software and numerical backend as generation.

~~~bash
python -m sster_data replay --out /data/sster --split train --shard SHARD_FROM_MANIFEST --id ID_FROM_MANIFEST
~~~

Generation freezes the width's ticket plan, matches focal tickets against seed-addressable seasonal metadata, simulates worlds and writes resumable physical chunks. It computes both channel marginal balances before packing that width. Committing each public/private shard and its checksummed receipt frees its temporary physical archive. Keep the build plans and receipts for verification. All three full splits require substantial disk capacity; the float32 values alone are approximately 0.93 TB for the unthinned training profile.
