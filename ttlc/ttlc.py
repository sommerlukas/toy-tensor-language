#!/usr/bin/env python3

import tempfile
import sys
import os
import subprocess
import argparse
import shutil

TTL_BINARY_DIR = "@CMAKE_RUNTIME_OUTPUT_DIRECTORY@"
LLVM_BINARY_DIR = "@LLVM_EXE_BINARY_DIR@"
LLVM_LIB_DIR = "@LLVM_LIB_BINARY_DIR@"


def run_command(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Compilation failed with error:")
        print(e.output.decode())
        print("Compilation error")
        sys.exit(-1)


class Compilation:

    def __init__(self, input_file):
        self._stages = []
        self._input_file = input_file

    def append(self, stage):
        self._stages.append(stage)

    def _iterate_stages(self, start_with, stop_before, stop_after, func):
        last_output = self._input_file
        started = False
        for step in self._stages:
            if not started and start_with and step.name() != start_with:
                continue
            started = True
            if step.name() == stop_before:
                break
            last_output = func(step, last_output)
            if step.name() == stop_after:
                break

        return last_output

    def show_steps(self, start_with, stop_before, stop_after):
        print("Compilation stages: ")
        self._iterate_stages(start_with, stop_before, stop_after,
                             lambda step, last_output: step.dump(last_output))

    def run(self, start_with, stop_before, stop_after):
        return self._iterate_stages(start_with, stop_before, stop_after,
                                    lambda step, last_output: step.run(last_output))


class Frontend:
    def __init__(self, verbose, save_temps):
        self._executable = os.path.join(TTL_BINARY_DIR, "ttl-cc")
        self._output_file = self._get_output_file(save_temps)

    def _get_output_file(self, save_temps):
        if save_temps:
            return os.path.join(save_temps, "codegen.mlir")

        return tempfile.NamedTemporaryFile(prefix="codegen_",
                                           suffix=".mlir", delete=True).name

    def _get_command(self, stage_input):
        command = [self._executable,
                   stage_input,
                   "-o", self._output_file]

        return command

    def name(self):
        return "frontend"

    def dump(self, stage_input):
        print(f"{self.name()}: {' '.join(self._get_command(stage_input))}")
        return self._output_file

    def run(self, stage_input):
        run_command(self._get_command(stage_input))
        return self._output_file


class MLIRStage:
    def __init__(self, stage_name, passes, verbose, save_temps):
        self._executable = os.path.join(TTL_BINARY_DIR, "ttl-opt")
        self._stage_name = stage_name
        self._passes = passes
        self._output_file = self._get_output_file(save_temps, stage_name)

    def _get_output_file(self, save_temps, stage_name):
        if save_temps:
            return os.path.join(save_temps, f"after-{stage_name}.mlir")

        return tempfile.NamedTemporaryFile(prefix=f"after-{stage_name}_",
                                           suffix=".mlir", delete=True).name

    def _get_passes(self):
        if isinstance(self._passes, str):
            return [f"--{self._passes}"]

        if isinstance(self._passes, list):
            return [f"--{x}" for x in self._passes]

        raise Exception("Unsupported pass format")

    def _get_command(self, stage_input):
        command = [self._executable]
        command.extend(self._get_passes())
        command.extend([stage_input, "-o", self._output_file])
        return command

    def name(self):
        return self._stage_name

    def dump(self, stage_input):
        print(f"{self.name()}: {' '.join(self._get_command(stage_input))}")
        return self._output_file

    def run(self, stage_input):
        run_command(self._get_command(stage_input))
        return self._output_file


class MLIRTranslate:
    def __init__(self, verbose, save_temps):
        self._executable = os.path.join(LLVM_BINARY_DIR, "mlir-translate")
        self._output_file = self._get_output_file(save_temps)

    def _get_output_file(self, save_temps):
        if save_temps:
            return os.path.join(save_temps, "translated.ll")

        return tempfile.NamedTemporaryFile(prefix="translated_",
                                           suffix=".ll", delete=True).name

    def _get_command(self, stage_input):
        command = [self._executable,
                   "--mlir-to-llvmir",
                   stage_input,
                   "-o", self._output_file]

        return command

    def name(self):
        return "mlir-translate"

    def dump(self, stage_input):
        print(f"{self.name()}: {' '.join(self._get_command(stage_input))}")
        return self._output_file

    def run(self, stage_input):
        run_command(self._get_command(stage_input))
        return self._output_file


class Clang:
    def __init__(self, verbose, save_temps):
        self._executable = os.path.join(LLVM_BINARY_DIR, "clang")
        self._output_file = self._get_output_file(save_temps)

    def _get_output_file(self, save_temps):
        if save_temps:
            return os.path.join(save_temps, "linked.exe")

        return tempfile.NamedTemporaryFile(prefix="linked_",
                                           suffix=".exe", delete=True).name

    def _get_command(self, stage_input):
        command = [self._executable,
                   "-rpath", f"{LLVM_LIB_DIR}",
                   f"-L{LLVM_LIB_DIR}",
                   "-lmlir_runner_utils", "-lmlir_c_runner_utils",
                   stage_input,
                   "-o", self._output_file]

        return command

    def name(self):
        return "link"

    def dump(self, stage_input):
        print(f"{self.name()}: {' '.join(self._get_command(stage_input))}")
        return self._output_file

    def run(self, stage_input):
        run_command(self._get_command(stage_input))
        return self._output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ttlc",
        description="Toy Tensor Language (TTL) compiler driver")

    parser.add_argument("input_file")
    parser.add_argument("-o", "--output", metavar="<output file>",
                        default="a.out",
                        help="File to store compilation output to. "
                        "Use '-' to print output to stdout")
    parser.add_argument("--show-stages", action="store_true",
                        help="Show compilation stages")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show compilation stages, do not compile")
    parser.add_argument("--save-temps", metavar="<dir>",
                        help="Preserve all intermediate files by storing them in <dir>")
    parser.add_argument("--start-with", metavar="<stage>",
                        help="Start compilation with <stage>")
    parser.add_argument("--stop-after", metavar="<stage>",
                        help="Stop compilation after <stage>")
    parser.add_argument("--stop-before", metavar="<stage>",
                        help="Stop compilation before <stage>")

    args = parser.parse_args()

    verbose = False
    save_temps = args.save_temps

    input_file = os.path.abspath(args.input_file)
    if not os.path.isfile(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(-1)

    _, file_ext = os.path.splitext(input_file)
    starts_with = args.start_with
    # Skip the frontend when passed a file with '.mlir' file extension.
    if not starts_with and file_ext == ".mlir":
        starts_with = "ttl-to-tensor"

    compilation = Compilation(input_file)
    compilation.append(Frontend(verbose, save_temps))
    compilation.append(
        MLIRStage("ttl-to-tensor", "convert-ttl-to-tensor", verbose, save_temps))
    compilation.append(
        MLIRStage("ttl-to-linalg", "convert-ttl-to-linalg", verbose, save_temps))
    compilation.append(
        MLIRStage("ttl-to-scalar", "convert-ttl-to-scalar", verbose, save_temps))
    compilation.append(
        MLIRStage("canonicalize-after-ttl",
                  ["reconcile-unrealized-casts", "canonicalize"],
                  verbose, save_temps))
    compilation.append(MLIRStage("bufferize",
                                 ["one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
                                  "buffer-results-to-out-params=hoist-static-allocs",
                                  "buffer-deallocation-pipeline"],
                                 verbose, save_temps))
    compilation.append(
        MLIRStage("lower-tensor", ["convert-tensor-to-linalg"],
                  verbose, save_temps))
    compilation.append(
        MLIRStage("lower-linalg", ["convert-linalg-to-loops"],
                  verbose, save_temps))
    compilation.append(MLIRStage("lower-memref", ["expand-strided-metadata",
                       "memref-expand", "finalize-memref-to-llvm",
                                                  "lower-affine"],
                                 verbose, save_temps))
    compilation.append(
        MLIRStage("lower-scf", ["convert-scf-to-cf"], verbose, save_temps))
    compilation.append(MLIRStage("lower-scalar", ["convert-arith-to-llvm",
                                                  "convert-index-to-llvm",
                       "convert-cf-to-llvm", "convert-func-to-llvm"],
                                 verbose, save_temps))
    compilation.append(MLIRStage("canonicalize-after-lower",
                       ["reconcile-unrealized-casts", "canonicalize"],
        verbose, save_temps))
    compilation.append(MLIRTranslate(verbose, save_temps))
    compilation.append(Clang(verbose, save_temps))

    if args.show_stages or args.dry_run:
        compilation.show_steps(
            starts_with, args.stop_before, args.stop_after)

    if not args.dry_run:
        output_file = compilation.run(
            starts_with, args.stop_before, args.stop_after)

        if args.output == "-":
            with open(output_file) as f:
                print("\n\nCompilation output:\n")
                print(f.read())
        else:
            user_output_file = os.path.abspath(args.output)
            shutil.copy2(output_file, user_output_file)
