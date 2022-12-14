import os
from pathlib import Path
import argparse

import io
from contextlib import redirect_stdout
from fairmotion.data import amass, bvh
from fairmotion.ops import conversions, motion as motion_ops
from tqdm import tqdm
from glob import glob
import traceback
import bpy

def main(args):

    print(f"Loading SMPL model...")
    bm = amass.load_body_model(bm_path=args.bm_path)

    print("Converting motions...")
    motion_files = glob(args.motion_files)

    print(f"Found {len(motion_files)} motion files to convert.")

    pbar = tqdm(motion_files)

    Path(args.bvh_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fbx_output_dir).mkdir(parents=True, exist_ok=True)

    for motion_file in pbar:
        try:
            motion_name = Path(motion_file).stem

            pbar.set_description(f"Loading motion {motion_name}...")
            motion = amass.load(motion_file, bm=bm)
            motion = motion_ops.rotate(motion, conversions.Ax2R(conversions.deg2rad(-90)))

            pbar.set_description(f"Starting convertion of {motion_name}...")
            bvh_filepath = os.path.join(args.bvh_output_dir, motion_name + ".bvh")
            fbx_filepath = os.path.join(args.fbx_output_dir, motion_name + ".fbx")

            if not os.path.exists(bvh_filepath) or args.force_generate:
                pbar.set_description(f"Converting to BVH {motion_name}...")
                bvh.save(motion=motion, filename=bvh_filepath)
            else:
                print(f"BVH file already exists, skipping.")

            if not os.path.exists(fbx_filepath) or args.force_generate:
                pbar.set_description(f"Converting to FBX {motion_name}...")

                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    bpy.ops.import_anim.bvh(filepath=bvh_filepath, frame_start=1, update_scene_fps=True, update_scene_duration=True)

                    # Insert T-Pose frame at t=0 to have a reference for initializing the humanoid avatar in Unity
                    armature = bpy.data.objects[bpy.context.active_object.name]
                    bpy.context.scene.frame_start = 0
                    bpy.context.scene.frame_set(0)
                    bpy.ops.object.mode_set(mode='POSE', toggle=False)
                    bpy.ops.pose.select_all(action='SELECT')
                    bpy.ops.pose.transforms_clear()
                    for i, bone in enumerate(armature.data.bones):
                        bone.select = True
                        armature.pose.bones[i].keyframe_insert(data_path="location", frame=0)
                        armature.pose.bones[i].keyframe_insert(data_path="rotation_euler", frame=0)
                    
                    bpy.ops.export_scene.fbx(filepath=fbx_filepath, axis_forward='-Z', axis_up='Y', use_selection=True)

                    # Clear scene
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    bpy.ops.object.select_all(action='SELECT')
                    bpy.ops.object.delete(use_global=False)
                    bpy.ops.outliner.orphans_purge()
                    bpy.context.scene.frame_end = 1
            else:
                print(f"FBX file already exists, skipping.")

        except Exception:
            print(f"Error while converting mocap animation file {motion.name}", traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AMASS mocap to FBX"
    )
    parser.add_argument("-f", "--force_generate", action='store_true', help="Force generating files")
    parser.add_argument("--bm_path", type=str, help="Path the the SMPL model .npz or .pkl file")
    parser.add_argument("--motion_files", type=str)
    parser.add_argument("--bvh_output_dir", type=str, help="Output directory where the BVH files will be written")
    parser.add_argument("--fbx_output_dir", type=str, help="Output directory where the FBX files will be written")
    args = parser.parse_args()
    main(args)
