#!/usr/bin/env bash
# Install the git hooks in this clone.
#
#     bash scripts/installHooks.sh
#
# Git does not track hooks, so run this once on every machine that pushes main. The
# installed file is a two line shim, and `scripts/hooks/` holds the hook itself, so an
# edit to a hook needs no second install.
#
# One hooks directory serves the main checkout and every worktree, so one install
# covers them all.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
hooks="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks"

for name in pre-push; do
  source="scripts/hooks/$(echo "$name" | sed -E 's/-(.)/\U\1/g')"
  test -f "$source" || { echo "no such hook: $source" >&2; exit 1; }
  target="$hooks/$name"
  if [[ -e "$target" ]] && ! grep -q "scripts/hooks/" "$target"; then
    mv "$target" "$target.replaced"
    echo "kept the hook that was there as $target.replaced"
  fi
  cat > "$target" <<SHIM
#!/usr/bin/env bash
# Installed by scripts/installHooks.sh. The hook itself is $source.
hook="\$(git rev-parse --show-toplevel)/$source"
test -x "\$hook" || exit 0
exec "\$hook" "\$@"
SHIM
  chmod +x "$target" "$source"
  echo "installed $name -> $source"
done
