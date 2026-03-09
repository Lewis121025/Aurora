# 文件名: 练习_运行时并发.py (实操演示)
import threading
import time

# 模拟全系统唯一的一个账本，和保证安全的锁
event_log = []
system_lock = threading.Lock()

def simulate_concurrent_user(user_id: int):
    print(f"🚀 用户 {user_id} 正在疯狂冲刺想写日记...")
    
    # 【核心！】如果没有这行锁，大家就同时乱写；有了它，大家只能乖乖排队！
    with system_lock:
        print(f"  🟢 [锁被打开] 恭喜用户 {user_id} 挤进屋子！")
        # 模拟极其消耗时间的写入记录（Event Sourcing）
        time.sleep(0.5) 
        event_log.append(f"Event_ID_{user_id}: 这句话是永远存在的底单")
        print(f"  🔴 [锁被关闭] 用户 {user_id} 记录完毕，离开了屋子。\n")

# 制造 3 个人并发拥堵抢占资源！
# 他们几乎在同一毫秒开跑
threads = []
for i in range(1, 4):
    t = threading.Thread(target=simulate_concurrent_user, args=(i,))
    threads.append(t)
    t.start()

# 等待所有人跑完
for t in threads:
    t.join()

print("最终毫无错乱的系统底单流水账 (Event Sourcing):")
print(event_log)
