#!/usr/bin/env sh
# Stamp index.html's CSS/JS links with the current time, so browsers fetch fresh copies
# after a deploy instead of using GitHub Pages' 10-minute cache. Run before committing front-end changes.
set -e
cd "$(dirname "$0")/.."
v=$(date +%Y%m%d%H%M)
sed -i -E "s/(app\.(css|js))\?v=[^\"]*/\1?v=$v/" index.html
echo "assets stamped v=$v"
