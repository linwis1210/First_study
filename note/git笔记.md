- 若第一次连接，使用SSH密钥，打开Git Bash，输入以下命令：

```
 ssh-keygen -t rsa
```

- 复制生成的SSH密钥

```
clip < ~/.ssh/id_rsa.pub
```

- 打开github，在setting中添加SSH keys，克隆记得使用SSH

- ```
  git config --global user.name "linwis1210"
  git config --global user.email "277533694@qq.com"
  ```

  

- ```
  git clone 
  git init
  git add filename
  git add .
  git commit -m "balaba"
  git status
  git push -u origin main/master
  git pull
  
  # 添加标签
  git tag v1.0.0
  git push --tag
  
  # 删除标签
  git tag -d tagName
  git push origin :refs/tags/tagName 
  ```

