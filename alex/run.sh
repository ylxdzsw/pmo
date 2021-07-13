#!/bin/bash

deepspeed train.py --deepspeed_config=ds_config.json -p 4 --steps=200
