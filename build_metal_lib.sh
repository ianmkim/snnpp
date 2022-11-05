#!/bin/bash
xcrun -sdk macosx metal -c add.metal -o mylibrary.air
xcrun -sdk macosx metallib mylibrary.air -o default.metallib

mv default.metallib build

