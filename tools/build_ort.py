#!/usr/bin/env python3
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from its software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Build the ONNX Runtime library used by the Triton ORT backend.

Generates Dockerfile.ort via gen_ort_dockerfile.py, runs docker build to
produce the tritonserver_onnxruntime image, and optionally extracts
/opt/onnxruntime to a local directory for reuse with
TRITON_ONNXRUNTIME_ARTIFACTS_PATH (or build.py --ort-artifacts).

Typical usage:
    # Build ORT from local clone, extract artifacts
    cd onnxruntime_backend
    python3 tools/build_ort.py \\
        --base-image tritonserver_buildbase \\
        --ort-repo ~/dev/tritonserver/onnxruntime \\
        --artifacts-out ~/dev/tritonserver/ort_artifacts

    # Reuse an existing tritonserver_onnxruntime image (skip docker build)
    python3 tools/build_ort.py \\
        --reuse-image \\
        --artifacts-out ~/dev/tritonserver/ort_artifacts
"""

import argparse
import os
import pathlib
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
BACKEND_DIR = SCRIPT_DIR.parent
GEN_DOCKERFILE = SCRIPT_DIR / "gen_ort_dockerfile.py"
ORT_DOCKER_IMAGE = "tritonserver_onnxruntime"
DEFAULT_ORT_VERSION = "1.24.4"


def default_memory_limit() -> str:
    """Return 80% of system RAM rounded down to the nearest OS page boundary,
    expressed as a byte count string suitable for docker --memory."""
    page_size = os.sysconf("SC_PAGE_SIZE")          # bytes per page (typically 4096)
    phys_pages = os.sysconf("SC_PHYS_PAGES")        # total physical pages
    total_bytes = page_size * phys_pages
    limit_bytes = int(total_bytes * 0.8)
    # Round down to the nearest page boundary (already a multiple of page_size,
    # but make the intent explicit).
    limit_bytes = (limit_bytes // page_size) * page_size
    return str(limit_bytes)


def run(cmd, check=True, **kwargs):
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, check=False, **kwargs)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build ONNX Runtime for Triton and optionally extract artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        help="Base Docker image for the ORT build. Must have CUDA, cuDNN and TensorRT "
        "(e.g. nvcr.io/nvidia/tritonserver:26.02-py3-min or tritonserver_buildbase). "
        "Required unless --reuse-image is set.",
    )
    parser.add_argument(
        "--ort-version",
        type=str,
        default=DEFAULT_ORT_VERSION,
        help=f"ORT version to build (default: {DEFAULT_ORT_VERSION}).",
    )
    parser.add_argument(
        "--ort-repo",
        type=str,
        default=None,
        help="Path to a local ORT git clone. Passed to docker build as a BuildKit "
        "named context instead of cloning from GitHub. "
        "Any fixes should be committed directly to this clone.",
    )
    parser.add_argument(
        "--artifacts-out",
        type=str,
        default=None,
        help="After a successful build, extract /opt/onnxruntime from the Docker image "
        "to this host directory. The directory is created if it does not exist. "
        "The result can be passed to build.py as --ort-artifacts.",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="/tmp/ort_docker_build",
        help="Working directory for the generated Dockerfile.ort (default: /tmp/ort_docker_build).",
    )
    parser.add_argument(
        "--ort-tensorrt",
        action="store_true",
        default=True,
        help="Enable TensorRT execution provider in ORT (default: on).",
    )
    parser.add_argument(
        "--no-ort-tensorrt",
        dest="ort_tensorrt",
        action="store_false",
        help="Disable TensorRT execution provider.",
    )
    parser.add_argument(
        "--trt-version",
        type=str,
        default=os.environ.get("TRT_VERSION", ""),
        help="TensorRT version string forwarded to gen_ort_dockerfile.py "
        "(defaults to $TRT_VERSION env var).",
    )
    parser.add_argument(
        "--min-compute-capability",
        type=str,
        default=None,
        help="Minimum CUDA compute capability (e.g. 8.0). "
        "Architectures below this are excluded from the build.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for ORT compilation inside Docker "
        "(passed as PARALLEL_JOBS build-arg).",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        help="Memory limit for the docker build (e.g. '16g', '8192m', or a raw byte "
        "count). Defaults to 80%% of system RAM rounded down to the nearest OS page "
        "boundary.",
    )
    parser.add_argument(
        "--reuse-image",
        action="store_true",
        default=False,
        help="Skip docker build and use the existing local tritonserver_onnxruntime image. "
        "Useful to only re-extract artifacts without rebuilding.",
    )
    parser.add_argument(
        "--ort-build-config",
        type=str,
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo"],
        help="ORT build configuration (default: Release).",
    )
    parser.add_argument(
        "--cudnn-home",
        type=str,
        default="/usr",
        help="cuDNN installation directory inside the base image (default: /usr).",
    )

    FLAGS = parser.parse_args()

    if not FLAGS.reuse_image and FLAGS.base_image is None:
        parser.error("--base-image is required unless --reuse-image is set.")

    build_dir = pathlib.Path(FLAGS.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    dockerfile_path = build_dir / "Dockerfile.ort"

    if not FLAGS.reuse_image:
        # --- Generate Dockerfile.ort ---
        gen_cmd = [
            sys.executable,
            str(GEN_DOCKERFILE),
            f"--triton-container={FLAGS.base_image}",
            f"--ort-version={FLAGS.ort_version}",
            f"--ort-build-config={FLAGS.ort_build_config}",
            "--enable-gpu",
            f"--cudnn-home={FLAGS.cudnn_home}",
            f"--output={dockerfile_path}",
        ]
        if FLAGS.ort_tensorrt:
            gen_cmd.append("--ort-tensorrt")
        if FLAGS.trt_version:
            gen_cmd.append(f"--trt-version={FLAGS.trt_version}")
        if FLAGS.min_compute_capability:
            gen_cmd.append(f"--min-compute-capability={FLAGS.min_compute_capability}")
        if FLAGS.ort_repo:
            gen_cmd.append("--ort-local-repo")

        run(gen_cmd)

        # --- docker build ---
        memory_limit = FLAGS.memory if FLAGS.memory else default_memory_limit()
        docker_cmd = [
            "docker",
            "build",
            "--load",
            "--memory", memory_limit,
            "-t",
            ORT_DOCKER_IMAGE,
            "-f",
            str(dockerfile_path),
        ]
        if FLAGS.jobs:
            docker_cmd += [f"--build-arg=PARALLEL_JOBS={FLAGS.jobs}"]
        if FLAGS.ort_repo:
            ort_repo_abs = os.path.abspath(FLAGS.ort_repo)
            docker_cmd += ["--build-context", f"ort_source={ort_repo_abs}"]
        # Build context is the onnxruntime_backend root (COPY custom_op_gbeausire etc.)
        docker_cmd.append(str(BACKEND_DIR))

        run(docker_cmd)

    # --- Extract artifacts ---
    if FLAGS.artifacts_out:
        artifacts_out = os.path.abspath(FLAGS.artifacts_out)
        print(f"\nExtracting /opt/onnxruntime from {ORT_DOCKER_IMAGE} to {artifacts_out}",
              flush=True)
        run(["docker", "rm", "-f", "ort_extract_tmp"], check=False)
        run(["docker", "create", "--name", "ort_extract_tmp", ORT_DOCKER_IMAGE])
        pathlib.Path(artifacts_out).parent.mkdir(parents=True, exist_ok=True)
        run(["docker", "cp", f"ort_extract_tmp:/opt/onnxruntime", artifacts_out])
        run(["docker", "rm", "ort_extract_tmp"], check=False)
        print(f"\nArtifacts extracted to {artifacts_out}", flush=True)
        print(
            f"Pass --ort-artifacts {artifacts_out} to server/build.py to reuse them.",
            flush=True,
        )


if __name__ == "__main__":
    main()
