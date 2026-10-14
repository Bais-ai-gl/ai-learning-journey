# Day 1 · 脚本 5：异常处理
# 练习：try/except/finally、自定义异常、raise

class ValueTooLargeError(Exception):     # 自定义异常
    pass

def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "错误：除数不能为0"
    except TypeError:
        return "错误：类型不正确"
    finally:
        print(f"  调用记录：safe_divide({a}, {b})")
    return result

def check_size(n):
    if n > 100:
        raise ValueTooLargeError(f"{n} 太大了")

print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide(10, "a"))

try:
    check_size(200)
except ValueTooLargeError as e:
    print("捕获到自定义异常：", e)
