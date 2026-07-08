from __future__ import annotations

import os
import pdb  # noqa:F401
import random
import re
from dataclasses import MISSING

import h5py
import isaaclab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt


# TODO: Sim folder cleanup
def spawn_multi_usd(
    prim_path: str,
    cfg: MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    prim_path = str(prim_path)
    # check prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # manually clone prims if the source prim path is a regex expression
    for i, prim_path in enumerate(prim_paths):
        # spawn asset from the selected usd file
        # usd_path = random.choice(cfg.usd_paths)  # random
        usd_path = cfg.usd_paths[i % len(cfg.usd_paths)]  # sequence

        asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=cfg.scale, rigid_props=cfg.rigid_props)
        prim = _spawn_from_usd_file(prim_path, usd_path, asset_cfg, translation, orientation)

        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # return the prim
    return prim


def spawn_multi_urdf(
    prim_path: str,
    cfg: MultiUrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    prim_path = str(prim_path)
    # check prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # container to save mesh name per environment
    env_id_to_obj_path = dict()

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # manually clone prims if the source prim path is a regex expression
    for i, prim_path in enumerate(prim_paths):
        # spawn asset from the selected usd file (urdf -> usd)
        # urdf_path = random.choice(cfg.urdf_paths)  # random
        urdf_path = cfg.urdf_paths[i % len(cfg.urdf_paths)]  # sequence
        usd_path = f"{'/'.join(urdf_path.split('/')[:-1])}/usd/{urdf_path.split('/')[-2]}.usd"

        env_id_to_obj_path[i] = usd_path

        if os.path.exists(usd_path):
            asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=cfg.scale, rigid_props=cfg.rigid_props)
        else:
            asset_cfg = sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                usd_dir=f"{'/'.join(urdf_path.split('/')[:-1])}/usd",
                usd_file_name=urdf_path.split("/")[-2] + ".usd",
                make_instanceable=cfg.make_instanceable,
                fix_base=cfg.fix_base,
                convex_decompose_mesh=cfg.convex_decompose_mesh,
                scale=cfg.scale,
                rigid_props=cfg.rigid_props,
            )
            urdf_loader = sim_utils.UrdfConverter(asset_cfg)
            usd_path = urdf_loader.usd_path

        prim = _spawn_from_usd_file(prim_path, usd_path, asset_cfg, translation, orientation)

        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # save mesh name per environment
    with h5py.File("env_id_to_obj_path.h5", "w") as hf:
        for k, v in env_id_to_obj_path.items():
            hf[str(k)] = v

    # return the prim
    return prim


def spawn_multi_urdf_sdf(
    prim_path: str,
    cfg: MultiUrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    prim_path = str(prim_path)
    # check prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    prim_utils.create_prim("/World/Dataset", "Scope")
    proto_prim_paths = list()
    for index, urdf_path in enumerate(cfg.urdf_paths):
        # spawn single instance
        proto_prim_path = f"/World/Dataset/Object_{index:02d}"
        usd_path = f"{'/'.join(urdf_path.split('/')[:-1])}/usd/{urdf_path.split('/')[-2]}.usd"
        if os.path.exists(usd_path):
            asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=cfg.scale, rigid_props=cfg.rigid_props)
        else:
            asset_cfg = sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                usd_dir=f"{'/'.join(urdf_path.split('/')[:-1])}/usd",
                usd_file_name=urdf_path.split("/")[-2] + ".usd",
                make_instanceable=cfg.make_instanceable,
                fix_base=cfg.fix_base,
                convex_decompose_mesh=cfg.convex_decompose_mesh,
                scale=cfg.scale,
                rigid_props=cfg.rigid_props,
            )
            urdf_loader = sim_utils.UrdfConverter(asset_cfg)
            usd_path = urdf_loader.usd_path

        prim = _spawn_from_usd_file(proto_prim_path, usd_path, asset_cfg, translation, orientation)

        # save the proto prim path
        proto_prim_paths.append(proto_prim_path)

        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # convert orientation ordering (wxyz to xyzw)
    orientation = (orientation[1], orientation[2], orientation[3], orientation[0])
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # randomly select an asset configuration
            # proto_path = random.choice(proto_prim_paths)  # random
            proto_path = proto_prim_paths[i % len(proto_prim_paths)]  # sequence

            # inherit the proto prim
            # env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))
            # set the translation and orientation
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()

            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*translation)

            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = Gf.Quatd(*orientation)

            scale_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            if scale_spec is None:
                scale_spec = Sdf.AttributeSpec(env_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale_spec.default = Gf.Vec3d(*cfg.scale)

            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            # # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # # Note: Just need to acquire the right attribute about the property you want to set
            # # Here is an example on setting color randomly
            # color_spec = env_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            # color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())

    # delete the dataset prim after spawning
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


def spawn_multi_urdf_sdf_w_random_scale(
    prim_path: str,
    cfg: MultiUrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    prim_path = str(prim_path)
    # check prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    prim_utils.create_prim("/World/Dataset", "Scope")
    proto_prim_paths = list()
    for index, urdf_path in enumerate(cfg.urdf_paths):
        # spawn single instance
        proto_prim_path = f"/World/Dataset/Object_{index:02d}"
        usd_path = f"{'/'.join(urdf_path.split('/')[:-1])}/usd/{urdf_path.split('/')[-2]}.usd"
        if os.path.exists(usd_path):
            asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=cfg.scale, rigid_props=cfg.rigid_props)
        else:
            asset_cfg = sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                usd_dir=f"{'/'.join(urdf_path.split('/')[:-1])}/usd",
                usd_file_name=urdf_path.split("/")[-2] + ".usd",
                make_instanceable=cfg.make_instanceable,
                fix_base=cfg.fix_base,
                convex_decompose_mesh=cfg.convex_decompose_mesh,
                scale=cfg.scale,
                rigid_props=cfg.rigid_props,
            )
            urdf_loader = sim_utils.UrdfConverter(asset_cfg)
            usd_path = urdf_loader.usd_path

        prim = _spawn_from_usd_file(proto_prim_path, usd_path, asset_cfg, translation, orientation)

        # save the proto prim path
        proto_prim_paths.append(proto_prim_path)

        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # convert orientation ordering (wxyz to xyzw)
    orientation = (orientation[1], orientation[2], orientation[3], orientation[0])
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            # randomly select an asset configuration
            # proto_path = random.choice(proto_prim_paths)  # random
            proto_path = proto_prim_paths[i % len(proto_prim_paths)]  # sequence

            # inherit the proto prim
            # env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))
            # set the translation and orientation
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()

            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*translation)

            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = Gf.Quatd(*orientation)

            scale_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            if scale_spec is None:
                scale_spec = Sdf.AttributeSpec(env_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
            sampled_scale = random.uniform(0.5, 1.0)
            sampled_scale = tuple([sampled_scale * cfg.scale[i] for i in range(3)])
            scale_spec.default = Gf.Vec3d(*sampled_scale)

            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            # # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # # Note: Just need to acquire the right attribute about the property you want to set
            # # Here is an example on setting color randomly
            # color_spec = env_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            # color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())

    # delete the dataset prim after spawning
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


def spawn_multi_object_randomly(
    prim_path: str,
    cfg: MultiAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # manually clone prims if the source prim path is a regex expression
    for prim_path in prim_paths:
        # randomly select an asset configuration
        asset_cfg = random.choice(cfg.assets_cfg)
        # spawn single instance
        prim = asset_cfg.func(prim_path, asset_cfg, translation, orientation)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # return the prim
    return prim


@configclass
class MultiUsdFileCfg(FileCfg):
    """Configuration parameters for loading multiple usd assets randomly."""

    func: sim_utils.SpawnerCfg.func = spawn_multi_usd
    usd_paths: list[str] = MISSING


@configclass
class MultiUrdfFileCfg(FileCfg, sim_utils.UrdfConverterCfg):
    """Configuration parameters for loading multiple urdf assets randomly."""

    func: sim_utils.SpawnerCfg.func = spawn_multi_urdf
    # func: sim_utils.SpawnerCfg.func = spawn_multi_urdf_sdf
    # func: sim_utils.SpawnerCfg.func = spawn_multi_urdf_sdf_w_random_scale
    urdf_paths: list[str] = MISSING


@configclass
class MultiAssetCfg(sim_utils.SpawnerCfg):
    """Configuration parameters for loading multiple assets randomly."""

    func: sim_utils.SpawnerCfg.func = spawn_multi_object_randomly
    assets_cfg: list[sim_utils.SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""


from collections.abc import Callable

################################
from typing import Dict

from isaaclab.sim import CuboidCfg, spawn_cuboid


def spawn_random_cuboid(
    prim_path: str,
    cfg: RandomCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """
    Create random size cuboids
    """
    axes = ["x", "y", "z"]
    size = []
    for id, ax in enumerate(axes):
        assert ax in cfg.size_range.keys()
        assert len(cfg.size_range[ax]) == 2
        size.append(random.uniform(cfg.size_range[ax][0], cfg.size_range[ax][1]))
    size = tuple(size)
    cfg.size = size
    prim = spawn_cuboid(prim_path, cfg, translation, orientation)
    return prim


@configclass
class RandomCuboidCfg(CuboidCfg):
    func: Callable = spawn_random_cuboid
    size_range: Dict[str, tuple[float, float]] = MISSING
