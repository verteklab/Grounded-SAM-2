"""
线程池性能压测脚本
对比线程池模式 vs 传统锁模式的性能差异
"""

import requests
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import statistics


def send_request(server_url, image_base64, request_id, timeout=300):
    """发送推理请求"""
    data = {
        'image_base64': image_base64,
        'text_prompt': 'road surface.',
        'box_threshold': 0.2,
        'text_threshold': 0.25,
        'epsilon': 1.0
    }
    
    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{server_url}/semantic-segmentation",
            json=data,
            timeout=timeout
        )
        duration = time.perf_counter() - start_time
        
        result = {
            'request_id': request_id,
            'status_code': response.status_code,
            'duration': duration,
            'success': response.status_code == 200,
            'error': None
        }
        
        if response.status_code == 200:
            try:
                result_data = response.json()
                result['success'] = result_data.get('status') == 'success'
                result['count'] = result_data.get('count', 0)
            except:
                result['error'] = '响应解析失败'
        else:
            result['error'] = f'HTTP {response.status_code}'
        
        return result
    except Exception as e:
        duration = time.perf_counter() - start_time
        return {
            'request_id': request_id,
            'status_code': 0,
            'duration': duration,
            'success': False,
            'error': str(e)
        }


def load_test_image(file_path):
    """加载测试图片"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return None


def run_benchmark(server_url, image_base64, num_requests=20, concurrent=10):
    """运行压测"""
    print("=" * 60)
    print(f"性能压测")
    print("=" * 60)
    print(f"服务器: {server_url}")
    print(f"总请求数: {num_requests}")
    print(f"并发数: {concurrent}")
    print()
    
    # 健康检查
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ 服务不健康: {health.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return
    
    print("✓ 服务健康检查通过")
    print()
    
    # 获取线程池状态
    try:
        pool_metrics = requests.get(f"{server_url}/pool-metrics", timeout=5)
        if pool_metrics.status_code == 200:
            metrics_data = pool_metrics.json()
            pool_enabled = metrics_data.get('thread_pool_enabled', False)
            print(f"线程池状态: {'启用' if pool_enabled else '禁用（传统锁模式）'}")
            if pool_enabled:
                config = metrics_data.get('config', {})
                print(f"  解码线程: {config.get('decode_threads', 'N/A')}")
                print(f"  预处理线程: {config.get('preprocess_threads', 'N/A')}")
            print()
    except:
        print("⚠️  无法获取线程池状态（可能未启用）")
        print()
    
    # 执行压测
    print("开始压测...")
    start_time = time.perf_counter()
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {
            executor.submit(send_request, server_url, image_base64, f"req_{i+1}"): i+1
            for i in range(num_requests)
        }
        
        completed = 0
        for future in as_completed(futures):
            request_num = futures[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result['success'] else "✗"
                print(f"[{completed}/{num_requests}] {status} 请求#{request_num} | "
                      f"Duration={result['duration']:.3f}s | "
                      f"Status={result['status_code']}")
                if not result['success']:
                    print(f"     错误: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"[{completed}/{num_requests}] ✗ 请求#{request_num} | 异常: {e}")
                results.append({
                    'request_id': f"req_{request_num}",
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })
    
    total_duration = time.perf_counter() - start_time
    
    # 统计分析
    print()
    print("=" * 60)
    print("压测结果统计")
    print("=" * 60)
    
    success_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"总请求数: {len(results)}")
    print(f"成功: {len(success_results)} ({len(success_results)/len(results)*100:.1f}%)")
    print(f"失败: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
    print()
    
    if success_results:
        durations = [r['duration'] for r in success_results]
        print(f"耗时统计（仅成功请求）:")
        print(f"  最短: {min(durations):.3f} 秒")
        print(f"  最长: {max(durations):.3f} 秒")
        print(f"  平均: {statistics.mean(durations):.3f} 秒")
        if len(durations) > 1:
            print(f"  中位数: {statistics.median(durations):.3f} 秒")
            print(f"  标准差: {statistics.stdev(durations):.3f} 秒")
        print(f"  总耗时: {total_duration:.3f} 秒")
        print(f"  吞吐量: {len(success_results)/total_duration:.2f} 请求/秒")
        print()
        
        # P50, P90, P99
        sorted_durations = sorted(durations)
        p50_idx = int(len(sorted_durations) * 0.5)
        p90_idx = int(len(sorted_durations) * 0.9)
        p99_idx = int(len(sorted_durations) * 0.99)
        print(f"延迟分位数:")
        print(f"  P50: {sorted_durations[p50_idx]:.3f} 秒")
        print(f"  P90: {sorted_durations[p90_idx]:.3f} 秒")
        print(f"  P99: {sorted_durations[p99_idx]:.3f} 秒")
        print()
    
    # 获取最终指标
    try:
        pool_metrics = requests.get(f"{server_url}/pool-metrics", timeout=5)
        if pool_metrics.status_code == 200:
            metrics_data = pool_metrics.json()
            metrics = metrics_data.get('metrics', {})
            print("线程池指标:")
            print(f"  任务提交: {metrics.get('tasks_submitted', 0)}")
            print(f"  任务完成: {metrics.get('tasks_completed', 0)}")
            print(f"  任务失败: {metrics.get('tasks_failed', 0)}")
            print(f"  任务超时: {metrics.get('tasks_timeout', 0)}")
            print(f"  任务拒绝: {metrics.get('tasks_rejected', 0)}")
            print(f"  成功率: {metrics.get('success_rate', 0)*100:.1f}%")
            print(f"  平均延迟: {metrics.get('avg_total_latency', 0):.3f} 秒")
            print(f"  解码队列长度: {metrics.get('decode_queue_current_size', 0)}")
            print(f"  推理队列长度: {metrics.get('inference_queue_current_size', 0)}")
            print()
    except:
        pass
    
    print("=" * 60)
    print("压测完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='线程池性能压测工具')
    parser.add_argument('--server', type=str, default='http://localhost:6155',
                       help='服务器地址 (默认: http://localhost:6155)')
    parser.add_argument('--image', type=str, required=True,
                       help='Base64图像文件路径')
    parser.add_argument('--requests', type=int, default=20,
                       help='总请求数 (默认: 20)')
    parser.add_argument('--concurrent', type=int, default=10,
                       help='并发数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 加载测试图片
    image_base64 = load_test_image(args.image)
    if not image_base64:
        print(f"❌ 无法加载测试图片: {args.image}")
        return
    
    # 运行压测
    run_benchmark(args.server, image_base64, args.requests, args.concurrent)


if __name__ == '__main__':
    main()

