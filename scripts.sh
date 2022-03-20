#!/bin/sh

# python -m run_session --dataset tmall --model GRU4Rec --split_type slices
# python -m run_session --dataset Tmall-session --model NARM --split_type slices
# python -m run_session --dataset Tmall-session --model STAMP --split_type slices
# python -m run_session --dataset Tmall-session --model SRGNN --split_type slices

# python -m run_session --dataset diginetica --model GRU4Rec --split_type slices
python -m run_session --dataset diginetica --model NARM --split_type slices
python -m run_session --dataset diginetica --model STAMP --split_type slices
python -m run_session --dataset diginetica --model SRGNN --split_type slices