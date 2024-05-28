# 請閱讀

clone 下來後請依需求以及實際情況修改 env 環境變數

## main.py

若 qdrant 容器所在的裝置為遠端裝置,請將<base_url>改成"ssh://<remote_username>@<remote_ipv4>"
若容器所在的裝置是本地的話,請依作業系統版本去做修改

- linux<br>
  以下範例為非 root 用戶的 docker desktop 的舉例
  ```
    unix:///home/{os.getlogin()}/.docker/desktop/docker.sock
  ```
- windows
  請參考[這裡](https://docs.docker.com/reference/cli/dockerd/#bind-docker-to-another-hostport-or-a-unix-socket)
  並且將 requests 套件版本指定為 2.31.0

  <br><br>

```python
if __name__ == "__main__":
    utils.Functional.start_qdrant_db(base_url=<base_url>)
    demo.launch(server_port=7861)
```

---

檔案轉換 目前僅支援 ppt, pptx, pdf, xlsx, csv 格式

xlsx 可能因為排版等等原因導致轉換成 pdf 的過程中遺漏一些資訊

---

=v=
