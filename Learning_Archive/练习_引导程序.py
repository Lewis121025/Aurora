# 文件名: 练习_引导程序.py (实操演示)

# 模拟一个会出问题的第三方云端连接
def connect_to_cloud(fake_password: str):
    print(f"  [尝试使用密码 '{fake_password}' 连接云端...]")
    if fake_password != "123456":
        # 模拟云端主动拒绝，抛出了一个我们不认识的可怕底层网络报错
        raise ConnectionError("远程服务器 403 Forbidden 拒绝了你的连接！")
    print("  [连接云端成功！]")
    return "云端记忆核心"

# 这是一个稳妥的工厂装配函数
def safe_bootstrap_engine(password: str):
    print("\n=== 开始装配记忆引擎 ===")
    try:
        # 我们把危险的动作包裹在 Try 里
        engine = connect_to_cloud(password)
        return engine
    except Exception as e:
        # 即使发生了最糟的网络爆炸，我们接住它，并且包装成一句话返回
        print(f"!!! 致命装配错误被拦截: {e} !!!")
        print("-> 自动降级：启动本地备用 Mock 引擎 (避免全系统死机)")
        return "本地 Mock 引擎"

# --- 开始测试 ---
print("第一轮：输入正确的密钥")
good_engine = safe_bootstrap_engine("123456")
print(f"最终系统运转依赖于: {good_engine}")

print("\n第二轮：输入被人篡改的密钥（感受防撞火墙的威力）")
bad_engine = safe_bootstrap_engine("wrong_key")
print(f"虽然爆炸了，但最终系统依然正常运转，只不过依赖于降级的: {bad_engine}")
