#!/usr/bin/env python3
"""
快速测试火山方舟 LLM + 阿里云百炼嵌入集成

使用方法:
    1. 确保已创建 .env 文件并填入 API Key
    2. 运行 python test_ark_integration.py
"""

import os
from aurora.config import AuroraSettings

# 验证配置是否已加载
settings = AuroraSettings()
if not settings.ark_api_key or not settings.bailian_api_key:
    # 尝试加载 .env (如果 pydantic-settings 没有自动加载)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        # 重新加载设置
        settings = AuroraSettings()
    except ImportError:
        pass
        
if not settings.ark_api_key:
    print("❌ 未找到 AURORA_ARK_API_KEY，请检查 .env 文件")
    exit(1)
if not settings.bailian_api_key:
    print("❌ 未找到 AURORA_BAILIAN_API_KEY，请检查 .env 文件")
    exit(1)

print(f"当前配置:")
print(f"- LLM: {settings.llm_provider} ({settings.ark_llm_model})")
print(f"- Embed: {settings.embedding_provider} ({settings.bailian_embedding_model})")



def test_llm_provider():
    """测试 LLM 提供者"""
    print("=" * 60)
    print("测试 LLM 提供者")
    print("=" * 60)
    
    from aurora.config import AuroraSettings
    from aurora.service import create_llm_provider
    from aurora.llm.schemas import PlotExtraction
    
    settings = AuroraSettings()
    print(f"LLM 提供者: {settings.llm_provider}")
    print(f"LLM 模型: {settings.ark_llm_model}")
    
    llm = create_llm_provider(settings)
    print(f"创建的提供者类型: {type(llm).__name__}")
    
    # 测试简单的 JSON 输出
    try:
        result = llm.complete_json(
            system="你是一个精确的信息提取器。",
            user="请分析以下对话:\nuser_message: 帮我写一首关于春天的诗\nagent_message: 好的，这是一首关于春天的诗...\n返回 PlotExtraction JSON。",
            schema=PlotExtraction,
            temperature=0.2,
            timeout_s=30.0,
        )
        print(f"✅ LLM 调用成功!")
        print(f"   动作: {result.action}")
        print(f"   参与者: {result.actors}")
        print(f"   情感效价: {result.emotion_valence}")
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")

def test_embedding_provider():
    """测试嵌入提供者"""
    print("\n" + "=" * 60)
    print("测试嵌入提供者")
    print("=" * 60)
    
    from aurora.config import AuroraSettings
    from aurora.service import create_embedding_provider
    
    settings = AuroraSettings()
    print(f"嵌入提供者: {settings.embedding_provider}")
    print(f"嵌入模型: {settings.bailian_embedding_model}")
    
    embedding = create_embedding_provider(settings)
    print(f"创建的提供者类型: {type(embedding).__name__}")
    
    # 测试嵌入生成
    try:
        text = "春天来了，万物复苏。"
        result = embedding.embed(text)
        print(f"✅ 嵌入生成成功!")
        print(f"   文本: {text}")
        print(f"   向量维度: {len(result)}")
        print(f"   向量前5维: {[round(x, 4) for x in result[:5]]}")
    except Exception as e:
        print(f"❌ 嵌入生成失败: {e}")


def test_full_integration():
    """测试完整的记忆系统集成"""
    print("\n" + "=" * 60)
    print("测试完整记忆系统集成")
    print("=" * 60)
    
    from aurora.config import AuroraSettings
    from aurora.service import AuroraTenant
    import uuid
    import tempfile
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = AuroraSettings(data_dir=tmpdir)
        
        print(f"数据目录: {tmpdir}")
        print(f"使用 LLM: {settings.llm_provider}")
        
        try:
            # 创建租户
            tenant = AuroraTenant(user_id="test_user", settings=settings)
            print(f"✅ 租户创建成功!")
            print(f"   LLM 类型: {type(tenant.llm).__name__}")
            
            # 摄入一条交互
            result = tenant.ingest_interaction(
                event_id=str(uuid.uuid4()),
                session_id="test_session",
                user_message="今天天气真好，我想出去散步。",
                agent_message="是的，春天的天气确实宜人。散步是个不错的选择，可以放松心情。",
            )
            print(f"✅ 交互摄入成功!")
            print(f"   Plot ID: {result.plot_id}")
            print(f"   编码: {result.encoded}")
            print(f"   惊奇度: {result.surprise:.4f}")
            
            # 查询记忆
            query_result = tenant.query(text="天气和心情", k=3)
            print(f"✅ 记忆查询成功!")
            print(f"   命中数: {len(query_result.hits)}")
            for hit in query_result.hits[:2]:
                print(f"   - {hit.kind}: {hit.snippet[:50]}... (分数: {hit.score:.4f})")
            
            # 获取自我叙事
            narrative = tenant.get_self_narrative()
            print(f"✅ 自我叙事获取成功!")
            print(f"   身份声明: {narrative['identity_statement'][:50]}...")
            
        except Exception as e:
            print(f"❌ 集成测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("🚀 AURORA 火山方舟集成测试")
    print()
    
    test_llm_provider()
    test_embedding_provider()
    test_full_integration()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
