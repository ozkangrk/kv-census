
import sys, importlib.util
spec = importlib.util.spec_from_file_location("fetch_census", "/home/ozkangu/Desktop/research/kv-census/scripts/fetch_census.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)
fc.ORGS = ["deepseek-ai", "RWKV"]  # smoke subset
sys.exit(fc.main())
