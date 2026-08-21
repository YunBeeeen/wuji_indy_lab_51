# Vendored Wuji Hand 1 description

- Repository: `https://github.com/wuji-technology/wuji-description`
- Commit: `06e5f14cdd1d5fad0a666ca463a668bf609f9534`
- Upstream release: `v2026.8.14`
- Downloaded GitHub archive SHA256: `2c43b89ebe5a75851cfe87fe49be1406ef35ab56bda643c8fe8d6b535f996cf6`
- Product: Wuji Hand 1, right hand (`hand/body`), not Wuji Hand 2
- Vendored files: right-hand URDF, official MJCF, and referenced right-hand STL meshes

`hand/body/mjcf/right.xml` is the unmodified upstream model.
`hand/body/mjcf/right_with_tip_sites.xml` differs only by five massless query
sites named `finger1_tip` through `finger5_tip`.  Their positions are copied
from the matching URDF fixed tip-joint origins.

The official MJCF is the best available simulation model source.  Its
controller parameters are not evidence that the real firmware uses the same
controller equation or units.
