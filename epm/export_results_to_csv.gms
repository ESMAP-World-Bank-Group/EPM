$if not set OUTPUT_DIR $set OUTPUT_DIR output
* create output directory
$call /bin/sh -c "mkdir -p '%OUTPUT_DIR'"
$exit