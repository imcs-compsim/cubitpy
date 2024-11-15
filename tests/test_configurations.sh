#!/bin/bash

# Test the deletion of objects.
# Save stderr in variable.
python test_delete.py

# Exit status to fail gitlab pipeline.
ERRORSTRING=$(python test_delete.py 2>&1)
if [[ -z "$ERRORSTRING" ]]; then
    # if std error was empy, test was ok
    echo "Deletion test OK!"
else
    echo "Deletion test FAILED!"
    exit 1
fi
