set -eu
revision="$1"
mkdir /opt/hermes
if [ -f /opt/hermes-source-cache/source.tar ]; then
    test "$revision" = "86fa9e6ca1de7ee71132d2be0103f3e37d1fa872"
    tar -xf /opt/hermes-source-cache/source.tar -C /opt/hermes
else
    git -C /opt/hermes init
    git -C /opt/hermes remote add origin https://github.com/NousResearch/hermes-agent.git
    git -C /opt/hermes fetch --depth 1 origin "$revision"
    git -C /opt/hermes checkout --detach FETCH_HEAD
    test "$(git -C /opt/hermes rev-parse HEAD)" = "$revision"
fi
rm -rf /opt/hermes-source-cache
printf '%s\n' "$revision" > /opt/hermes/.arena-revision
cd /opt/hermes
uv sync --frozen --no-dev --extra ddgs
