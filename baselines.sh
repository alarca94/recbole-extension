#!/bin/sh

# python -m run_session --dataset tmall --model GRU4Rec
# python -m run_session --dataset Tmall-session --model NARM
# python -m run_session --dataset Tmall-session --model STAMP
# python -m run_session --dataset Tmall-session --model SRGNN

# python -m run_session --dataset diginetica --model GRU4Rec
# python -m run_session --dataset diginetica --model NARM
# python -m run_session --dataset diginetica --model STAMP
python -m run_session --dataset diginetica --model SRGNN
# python -m run_session --dataset diginetica --model DHAGNN