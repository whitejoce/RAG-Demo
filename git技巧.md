## 偷懒技巧（不用打那么多字）

```bash
# 设置别名，一键存档
git config --global alias.save '!git add -A && git commit -m "save: $(date +%H:%M:%S)"'

# 之后直接：
git save
# agent 改
# 不满意 → git reset --hard HEAD
```

## 一句话总结

**改前 commit，不舒服 reset --hard。** 只要 commit 做了，就有后悔药吃。

有没 commit 但改乱了的特殊情况吗？我可以教你怎样不丢改动地抢救。