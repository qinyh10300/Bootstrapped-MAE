

在 `bash` 中加载 YAML 文件并解析其中的参数并不是一件直接的事，因为 `bash` 本身并不支持直接解析 YAML 格式。

`yq` 是一个非常流行的命令行工具，可以用来处理 YAML 文件，详细内容可以参考[这个仓库](https://github.com/mikefarah/yq)。它类似于 `jq`（用于 JSON），但专门用于 YAML 格式。

```
sudo snap install yq
```

`yq` 在处理 YAML 文件时，实际上是依赖于 `jq` 来进行 JSON 处理的。

如果你没有安装`jq`，可以运行以下命令：

```
sudo apt-get install jq
```