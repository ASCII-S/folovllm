# isinstance 函数

## 简介
`isinstance` 是 Python 的内置函数，用于**检查对象是否是指定类型的实例**。

## 语法
```python
isinstance(object, classinfo)
```

## 参数
- `object`: 要检查的对象
- `classinfo`: 类型或类型元组

## 返回值
返回布尔值（`True` 或 `False`）

## 使用示例

### 单个类型检查
```python
isinstance("hello", str)        # True
isinstance(42, int)             # True
isinstance([1, 2], dict)        # False
```

### 多个类型检查（使用元组）
```python
isinstance(prompt, (str, list)) # prompt 是 str 或 list 则返回 True
isinstance(42, (int, float))    # True
```

### 在 llm_engine.py 中的用法
```python
if isinstance(prompt, str):
    # 如果 prompt 是字符串类型，执行相应处理
    pass
```

## 常见用途
1. **类型检查和验证** - 确保参数类型正确
2. **实现多态行为** - 根据不同类型执行不同逻辑
3. **支持继承** - 比 `type(obj) == str` 更好，因为支持子类检查

## 与 type() 的区别
```python
class A:
    pass

class B(A):
    pass

b = B()

isinstance(b, A)    # True - 支持继承
type(b) == A        # False - 只检查确切类型
```

**推荐使用 `isinstance` 而不是 `type()` 进行类型检查。**

