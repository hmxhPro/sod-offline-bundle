# 调试按钮变灰问题

## 检查步骤

打开浏览器开发者工具（F12），在 Console 中执行以下代码检查状态：

```javascript
// 1. 检查所有相关状态
console.log({
  target: document.querySelector('textarea[placeholder*="船只"]')?.value,
  videoFile: document.querySelector('input[type="file"][accept*="video"]')?.files?.[0]?.name,
  mode: document.querySelector('[class*="border-brand-400"]')?.textContent,
  submitting: document.querySelector('button[type="submit"]')?.disabled,
  notConfigured: document.querySelector('[class*="amber-50"]') ? true : false,
  buttonDisabled: document.querySelector('button:has([class*="Sparkles"])')?.disabled
})

// 2. 检查 canSubmit 的各个条件
const target = document.querySelector('textarea[placeholder*="船只"]')?.value?.trim() || '';
const hasVideo = document.querySelector('input[type="file"][accept*="video"]')?.files?.length > 0;
const notConfigured = document.querySelector('[class*="amber-50"]') !== null;
const submitting = document.querySelector('button[type="submit"]')?.disabled;

console.log('canSubmit 条件检查:', {
  'target 是否填写': target.length > 0,
  'videoFile 是否上传': hasVideo,
  'notConfigured': notConfigured,
  'submitting': submitting,
  'canSubmit 应该为': !submitting && !notConfigured && target.length > 0 && hasVideo
})
```

## 可能的原因

1. **target 输入框的值被清空了** - 检查是否点击模型后触发了某些副作用
2. **videoFile 状态丢失** - 检查 React 状态是否被意外重置
3. **notConfigured 变成 true** - 检查配置加载状态
4. **submitting 还是 true** - 检查是否有异步操作未完成

## 临时解决方案

如果确认是状态问题，可以尝试：
1. 切换模型后，重新填写目标描述
2. 切换模型后，重新选择视频文件
3. 刷新页面后再试
