# isaac_collision/ — Isaac 이 실제로 쓰는 충돌 메시

여기 있는 STL 은 **공식 description 의 일부가 아니다.** 출처는

    nrmk_isaaclab_wuji/isaac_neuromeka/assets/model/urdf/wuji_right/meshes/

즉 Isaac 이 USD 로 임포트한 그 로컬 URDF 의 `*_collision.STL` 이다.
공식 description 트리(`wuji_description/`)는 커밋으로 pin 되어 있으므로
그 안에 섞지 않고 여기 따로 둔다.

## 왜 필요한가

두 시뮬 다 충돌 근사가 **convexHull** 이다 (Isaac 은
`configuration/wuji_right_physics.usd` 에 `convexHull`, MuJoCo 는
`mesh_graphadr >= 0`). 근사 방식은 같은데 **hull 을 만드는 원본 메시가
달랐다.**

| | 원본 | hull |
|---|---|---|
| Isaac | `palm_link_collision.STL` (500 삼각, 간략화) | PhysX convexHull |
| MuJoCo (기존) | `right_palm_link.STL` (9122 삼각, 렌더용) | 377 vert / 750 face |

정밀 메시의 hull 이 더 크게 부풀어서, Isaac 에서는 손 바깥인 스틱이
MuJoCo 에서는 `palm ↔ stick2 -3.94 mm` 관통으로 잡혔다.

같은 원본을 쓰면 두 hull 이 같아진다. 그래서 여기 복사해 두고
`right_with_tip_sites.xml` 의 손바닥 **충돌** geom 만 이걸 가리키게 했다.
**visual geom 은 공식 메시 그대로** 이므로 보이는 모양은 변하지 않는다.

## 갱신

로컬 URDF 의 충돌 메시가 바뀌면 여기도 다시 복사해야 한다. 자동 동기화는 없다.
