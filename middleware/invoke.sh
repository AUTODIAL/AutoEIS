#!/bin/sh
#
# jupyter
#
/usr/bin/invoke_app "$@" -t autoeis \
                        -C "start_jupyter -T @tool <AutoEIS_workflow.ipynb>" \
                        -u anaconda-7 \
                        -r none \
                        -w headless
