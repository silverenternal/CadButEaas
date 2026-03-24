# 前端 DXF 文件上传测试指南

## 问题诊断

### 症状
用户反馈前端上传 DXF 文件时出现"网络错误"。

### 可能的原因

1. **后端服务未运行**
   - 检查：访问 http://localhost:3000/health
   - 解决：`cargo run --package cad-cli -- serve --port 3000`

2. **API URL 配置错误**
   - 开发环境需要使用 Vite 代理
   - 生产环境需要 nginx 反向代理

3. **文件大小超限**
   - 后端限制：最大 50MB
   - 错误提示：HTTP 413

4. **CORS 跨域问题**
   - 检查浏览器控制台是否有 CORS 错误

## 前端配置检查

### 1. 开发环境（Vite）

确保 `.env.development` 文件存在：

```bash
VITE_API_URL=http://localhost:3000/api
VITE_WS_URL=ws://localhost:3000/ws
```

Vite 配置（`vite.config.ts`）已包含代理：

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:3000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
    '/ws': {
      target: 'ws://localhost:3000',
      ws: true,
    },
  },
}
```

### 2. 生产环境（Docker + nginx）

nginx 配置（`nginx.conf`）：

```nginx
location /api {
    proxy_pass http://backend:3000;
}

location /ws {
    proxy_pass http://backend:3000/ws;
}
```

## 测试步骤

### 方法 1：使用浏览器开发者工具

1. 打开浏览器开发者工具（F12）
2. 切换到 Network 标签
3. 点击"打开文件"按钮
4. 选择 DXF 文件
5. 观察请求：
   - URL: `http://localhost:3000/api/process` (开发环境)
   - Method: `POST`
   - Content-Type: `multipart/form-data`
   - 状态码：200 表示成功

### 方法 2：直接测试 API

```bash
# 健康检查
curl http://localhost:3000/health

# 直接上传（不使用前端）
curl -X POST http://localhost:3000/process \
  -F "file=@/path/to/file.dxf" \
  | jq '.edges | length'
```

### 方法 3：启动前端开发服务器

```bash
cd cad-web

# 安装依赖（首次）
pnpm install

# 启动开发服务器
pnpm dev

# 访问 http://localhost:5173
```

## 常见错误及解决方案

### 错误 1："无法连接到服务器"

**原因**：后端服务未运行

**解决**：
```bash
cargo run --package cad-cli -- serve --port 3000
```

### 错误 2："文件过大"

**原因**：文件超过 50MB 限制

**解决**：压缩 DXF 文件或联系管理员调整限制

### 错误 3："HTTP 404 Not Found"

**原因**：API URL 配置错误

**解决**：
- 检查 `.env.development` 文件
- 确认 VITE_API_URL 配置正确
- 重启 Vite 开发服务器

### 错误 4："CORS policy"

**原因**：跨域请求被阻止

**解决**：
- 确保使用 Vite 代理（开发环境）
- 确保使用 nginx 反向代理（生产环境）

## 前端代码修复

### 改进的错误处理

已更新 `api-client.ts` 中的错误处理：

```typescript
xhr.onerror = () => {
  if (xhr.status === 0) {
    reject(new NetworkError('无法连接到服务器，请检查后端服务是否运行'))
  } else if (xhr.status === 413) {
    reject(new NetworkError('文件过大，请上传小于 50MB 的文件'))
  } else if (xhr.status === 400) {
    reject(new NetworkError('请求格式错误'))
  } else if (xhr.status === 500) {
    reject(new NetworkError('服务器内部错误'))
  } else {
    reject(new NetworkError(`网络错误 (HTTP ${xhr.status})`))
  }
}
```

## 验收标准

- [ ] 前端可以正常选择 DXF 文件
- [ ] 上传进度条正常显示
- [ ] 小文件（<1MB）上传成功
- [ ] 中等文件（1-5MB）上传成功
- [ ] 大文件（5-50MB）上传成功
- [ ] 错误文件（>50MB）显示友好提示
- [ ] 中文文件名正常处理
- [ ] 网络错误显示友好提示

## 下一步

1. 启动后端服务
2. 启动前端开发服务器
3. 在浏览器中测试文件上传
4. 检查 Network 标签中的请求详情
