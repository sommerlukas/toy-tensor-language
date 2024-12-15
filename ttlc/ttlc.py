#!/usr/bin/env python3

import tempfile
import sys
import os
import subprocess
import argparse

TTL_BINARY_DIR = "@CMAKE_RUNTIME_OUTPUT_DIRECTORY@"
LLVM_BINARY_DIR = "@LLVM_EXE_BINARY_DIR@"


def run_command(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Compilation failed with error:")
        print(e.output.decode())
        print("Compilation error")
        sys.exit(-1)


class Compilation:

    def __init__(self):
        self._stages = []

    def append(self, stage):
        self._stages.append(stage)

    def show_steps(self):
        last_output = None
        print("Compilation stages: ")
        for step in self._stages:
            last_output = step.dump(last_output)

    def run(self):
        last_output = None
        for step in self._stages:
            last_output = step.run(last_output)

        return last_output


class Frontend:
    def __init__(self, input_file, verbose, save_temps):
        self._input_file = input_file
        self._executable = os.path.join(TTL_BINARY_DIR, "ttl-cc")
        self._output_file = self._get_output_file(save_temps)

    def _get_output_file(self, save_temps):
        if save_temps:
            return os.path.join(save_temps, "codegen.mlir")

        return tempfile.NamedTemporaryFile(prefix="codegen_",
                                           suffix=".mlir", delete=True).name

    def _get_command(self):
        command = [self._executable,
                   self._input_file,
                   "-o", self._output_file]

        return command

    def dump(self, stage_input):
        print(" ".join(self._get_command()))
        return self._output_file

    def run(self, stage_input):
        run_command(self._get_command())
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

    def dump(self, stage_input):
        print(" ".join(self._get_command(stage_input)))
        return self._output_file

    def run(self, stage_input):
        run_command(self._get_command(stage_input))
        return self._output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ttlc",
        description="Toy Tensor Language (TTL) compiler driver")

    parser.add_argument("input_file")
    parser.add_argument("--show-stages", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-temps")

    args = parser.parse_args()

    verbose = False
    save_temps = args.save_temps

    input_file = args.input_file
    if not os.path.isfile(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(-1)

    compilation = Compilation()
    compilation.append(Frontend(input_file, verbose, save_temps))
    compilation.append(
        MLIRStage("ttl-to-tensor", "convert-ttl-to-tensor", verbose, save_temps))
    compilation.append(
        MLIRStage("ttl-to-linalg", "convert-ttl-to-linalg", verbose, save_temps))
    compilation.append(
        MLIRStage("ttl-to-scalar", "convert-ttl-to-scalar", verbose, save_temps))
    compilation.append(
        MLIRStage("canonicalize", ["reconcile-unrealized-casts", "canonicalize"], verbose, save_temps))

    if args.show_stages or args.dry_run:
        compilation.show_steps()

    if not args.dry_run:
        output_file = compilation.run()

        with open(output_file) as f:
            print("\n\nCompilation output:\n")
            print(f.read())
