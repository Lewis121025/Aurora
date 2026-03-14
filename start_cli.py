import sys
import os
from datetime import datetime, timezone
import readline

from aurora.host_runtime.config import AuroraSettings
from aurora.host_runtime.runtime import AuroraRuntime

def main():
    print("==================================================")
    print(" Aurora v10.0 [Holistic Mind Architecture] Booting ")
    print("==================================================")
    print("Initializing Semantic Encoder (bge-small)...")
    
    # Try to load existing .env if present
    from pathlib import Path
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ[k] = v.strip().strip("'").strip('"')
                    
    if "AURORA_PROVIDER_API_KEY" not in os.environ:
        print("\n[WARNING] No AURORA_PROVIDER_API_KEY found in environment or .env file.")
        print("Aurora needs an API key to 'speak'.")
        key = input("Enter OpenAI-compatible API key: ").strip()
        os.environ["AURORA_PROVIDER_API_KEY"] = key
        
        url = input("Enter Base URL (default: https://api.openai.com/v1): ").strip()
        if url:
            os.environ["AURORA_PROVIDER_BASE_URL"] = url
            
        model = input("Enter Model name (default: gpt-4o-mini): ").strip()
        if model:
            os.environ["AURORA_PROVIDER_MODEL"] = model

    try:
        runtime = AuroraRuntime()
    except Exception as e:
        print(f"\nFailed to boot Aurora Runtime: {e}")
        sys.exit(1)
        
    print("\n[System Ready] You are now connected to Aurora's terminal.")
    print("Type '/health' to check status, or '/quit' to disconnect.")
    print("--------------------------------------------------\n")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if not user_input.strip():
                continue
                
            if user_input.strip() == "/quit":
                print("Disconnecting...")
                break
                
            if user_input.strip() == "/health":
                h = runtime.health()
                print(f"Health: alive={h.substrate_alive}, anchor_count={h.anchor_count}, provider_healthy={h.provider_healthy}")
                continue
                
            print("Aurora is thinking...", end='\r')
            sys.stdout.flush()
            
            outcome = runtime.handle_input(user_input)
            
            # Clear the "thinking" line
            print(" " * 30, end='\r')
            
            if outcome.outcome == "silence":
                print(f"[Aurora chose to remain silent. It's boundary/verbosity budgets determined this.]")
            else:
                print(f"Aurora: {outcome.output_text}")
                
        except KeyboardInterrupt:
            print("\nDisconnecting...")
            break
        except Exception as e:
            print(f"\n[Error during interaction]: {e}")

if __name__ == "__main__":
    # Ensure PYTHONPATH is set so aurora is importable
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    main()
