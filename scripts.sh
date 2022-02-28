#!/bin/sh

python -m run_session --dataset Tmall-session --model GRU4Rec
python -m run_session --dataset Tmall-session --model NARM
python -m run_session --dataset Tmall-session --model STAMP
python -m run_session --dataset Tmall-session --model SRGNN

python -m run_session --dataset mi-diginetica-session --model GRU4Rec
python -m run_session --dataset mi-diginetica-session --model NARM
python -m run_session --dataset mi-diginetica-session --model STAMP
python -m run_session --dataset mi-diginetica-session --model SRGNN