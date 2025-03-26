# src/logo_detector.py
import asyncio
import aiohttp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
import time

async def download_image_async(session: aiohttp.ClientSession, url: str) -> np.ndarray:
    """非同期で画像をダウンロード"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.read()
                image = np.asarray(bytearray(data), dtype="uint8")
                return cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:
                print(f"Failed to download image from {url}: {response.status}")
                return None
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

async def process_single_image(session: aiohttp.ClientSession, 
                             target_url: str, 
                             logos_dir: str,
                             image_id: str = None) -> Dict:
    """1つの画像に対するロゴ検出処理"""
    try:
        # 画像のダウンロード
        target_img_color = await download_image_async(session, target_url)
        if target_img_color is None:
            return {'url': target_url, 'error': 'Failed to download image', 'results': [], 'id': image_id}
        
        results = detect_multiple_logos_for_image(target_img_color, logos_dir)
        
        return {
            'url': target_url,
            'id': image_id,
            'results': results
        }
    except Exception as e:
        return {'url': target_url, 'error': str(e), 'results': [], 'id': image_id}

def detect_multiple_logos_for_image(target_img_color: np.ndarray, logos_dir: str) -> List[Dict]:
    """1つの画像に対する複数ロゴの検出（並列処理化）"""
    # 閾値の設定

    TEMPLATE_MATCHING_THRESHOLD = 0.4
    LOWE_RATIO_THRESHOLD = 0.7
    MIN_SCORE_THRESHOLD = 5.0
    
    try:
        # ロゴディレクトリから全ての画像を取得
        logo_files = [f for f in os.listdir(logos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not logo_files:
            raise ValueError(f"No logo images found in {logos_dir}")

        def process_logo(logo_file: str) -> Dict:
            try:
                logo_path = os.path.join(logos_dir, logo_file)
                logo_img_color = cv2.imread(logo_path)
                if logo_img_color is None:
                    print(f"Failed to load logo: {logo_file}")
                    return None

                # 画像の前処理
                target_img = preprocess_image(target_img_color)
                logo_img = preprocess_image(logo_img_color)

                # スケールの最適化
                scales = [0.15, 0.2, 0.25]  # スケール数を削減
                best_result = None
                max_score = 0

                for scale in scales:
                    # ロゴ画像をリサイズ
                    width = int(logo_img.shape[1] * scale)
                    height = int(logo_img.shape[0] * scale)
                    scaled_logo = cv2.resize(logo_img, (width, height), interpolation=cv2.INTER_CUBIC)
                    scaled_logo_color = cv2.resize(logo_img_color, (width, height), interpolation=cv2.INTER_CUBIC)

                    # テンプレートマッチング
                    result = cv2.matchTemplate(target_img, scaled_logo, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    if max_val > TEMPLATE_MATCHING_THRESHOLD:
                        # SIFT特徴点検出とマッチング
                        sift = cv2.SIFT_create(
                            nfeatures=0,
                            nOctaveLayers=5,
                            contrastThreshold=0.02,
                            edgeThreshold=10,
                            sigma=1.6
                        )

                        kp1, des1 = sift.detectAndCompute(scaled_logo, None)
                        kp2, des2 = sift.detectAndCompute(target_img, None)

                        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                            # FLANN matcher
                            FLANN_INDEX_KDTREE = 1
                            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                            search_params = dict(checks=50)
                            flann = cv2.FlannBasedMatcher(index_params, search_params)

                            matches = flann.knnMatch(des1, des2, k=2)

                            # Loweのratio test
                            good_matches = []
                            for m, n in matches:
                                if m.distance < LOWE_RATIO_THRESHOLD * n.distance:
                                    good_matches.append(m)

                            combined_score = max_val * len(good_matches)

                            if combined_score > max_score:
                                max_score = combined_score
                                result_img = target_img_color.copy()

                                if len(good_matches) >= 4:
                                    # 特徴点の座標を取得
                                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                                    # ホモグラフィ行列を計算
                                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                                    if H is not None:
                                        h, w = scaled_logo.shape[:2]
                                        logo_corners = np.float32([
                                            [0, 0],
                                            [0, h-1],
                                            [w-1, h-1],
                                            [w-1, 0]
                                        ]).reshape(-1, 1, 2)

                                        transformed_corners = cv2.perspectiveTransform(logo_corners, H)
                                        cv2.polylines(result_img, [np.int32(transformed_corners)], True, (0, 255, 0), 2)

                                matching_result = cv2.drawMatches(
                                    scaled_logo_color, kp1,
                                    target_img_color, kp2,
                                    good_matches[:10], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                )

                                best_result = {
                                    'logo_name': os.path.splitext(logo_file)[0],
                                    'score': combined_score,
                                    'scale': scale,
                                    'matches_count': len(good_matches),
                                    'result_image': result_img,
                                    'matching_visualization': matching_result
                                }

                                # 十分なスコアが得られた場合は早期リターン
                                if combined_score > MIN_SCORE_THRESHOLD:
                                    return best_result

                return best_result if max_score > MIN_SCORE_THRESHOLD else None

            except Exception as e:
                print(f"Error processing logo {logo_file}: {str(e)}")
                return None

        # 並列処理の実行
        all_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_logo = {executor.submit(process_logo, logo_file): logo_file 
                            for logo_file in logo_files}
            
            for future in as_completed(future_to_logo):
                result = future.result()
                if result is not None:
                    all_results.append(result)
                    # logo_name = os.path.splitext(future_to_logo[future])[0]  # 拡張子を除去
                    # print(f"{logo_name} detected with score {result['score']:.2f}")

        return all_results

    except Exception as e:
        print(f"Error during logo detection: {str(e)}")
        return []

async def process_multiple_images(image_data: List[Dict[str, str]], logos_dir: str):
    async with aiohttp.ClientSession() as session:
        tasks = []
        # CPU コア数に基づいて同時実行数を制限
        max_concurrent = os.cpu_count() or 2  # CPU コア数が取得できない場合は4
        max_concurrent =max_concurrent * 2  # CPU コア数が取得できない場合は4
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # バッチサイズを定義
        BATCH_SIZE = 40
        
        async def process_batch(batch_data):
            results = []
            for data in batch_data:
                async with semaphore:
                    result = await process_single_image(
                        session=session,
                        target_url=data['url'],
                        logos_dir=logos_dir,
                        image_id=data['id']
                    )
                    results.append(result)
            return results
        
        # バッチ処理の実装
        all_results = []
        for i in range(0, len(image_data), BATCH_SIZE):
            batch = image_data[i:i + BATCH_SIZE]
            results = await process_batch(batch)
            all_results.extend(results)
            
            # バッチ間で少し待機して、システムに余裕を持たせる
            await asyncio.sleep(0.1)
        
        return all_results

def display_results(all_results: List[Dict], show_images: bool = False) -> List[Dict]:
    """検出結果をJSON形式で返す"""
    output = []
    
    # 現在のUTC時刻を取得
    current_utc = time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    for image_result in all_results:
        if 'error' in image_result:
            continue
            
        # ロゴが検出された場合のみブランド情報を追加（スコアが3以上のもののみ）
        brands = [
            {
                'brand': result['logo_name'],
                'score': round(float(result['score']), 3)  # フォーマット方法を変更
            }
            for result in image_result['results']
            if result['score'] >= 3
        ]
        
        # 全ての画像に対して結果を出力
        sorted_brands = sorted(brands, key=lambda x: x['score'], reverse=True) if brands else []
        output.append({
            # 'ver': 1,
            'id': image_result['id'],
            'brands': brands if brands else None,
            'bestMatch': sorted_brands[0]['brand'] if sorted_brands else None,
            'status': 'DONE',
            'updated_at': current_utc
        });
        
        # 画像表示部分
        if show_images and brands:
            for result in image_result['results']:
                if result['score'] >= 3:
                    plt.figure(figsize=(15, 5))
                    
                    # 検出枠表示
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(result['result_image'], cv2.COLOR_BGR2RGB))
                    plt.title(f'Detection: {result["logo_name"]}\nScore: {result["score"]:.3g}, Scale: {result["scale"]:.2f}')
                    plt.axis('off')
                    
                    # 特徴点マッチングの表示
                    plt.subplot(1, 2, 2)
                    plt.imshow(cv2.cvtColor(result['matching_visualization'], cv2.COLOR_BGR2RGB))
                    plt.title(f'Feature Matching\nMatches: {result["matches_count"]}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
    
    return output

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """画像の前処理を行う"""
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ノイズ除去のためにガウシアンブラを適用
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

async def fetch_image_data_from_api(limit: int = 30, sellerId: str = ''):
    async with aiohttp.ClientSession() as session:  # ここでセッションを作成
        # グローバルIPの取得
        ip_response = await session.get('https://api.ipify.org?format=json')
        global_ip = (await ip_response.json())['ip']
        # print(f"現在のグローバルIP: {global_ip}")
        
        api_url = f"https://rex-server.f5.si/api/rex/inventory/logo-detection/get?limit={limit}&sellerId={sellerId}&ip={global_ip}"
        # api_url = f"http://localhost:3000/api/rex/inventory/logo-detection/get?limit={limit}&sellerId={sellerId}&ip={global_ip}" 
        
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                return [{'id': item['id'], 'url': item['url']} for item in data]
            else:
                raise Exception(f"API request failed with status {response.status}")

async def post_results_to_api(results: List[Dict]):
    """検出結果をAPIにPOST"""
    async with aiohttp.ClientSession() as session:
        # グローバルIPの取得
        ip_response = await session.get('https://api.ipify.org?format=json')
        global_ip = (await ip_response.json())['ip']
        api_url = f"https://rex-server.f5.si/api/rex/inventory/logo-detection/update-results?ip={global_ip}"

        async with session.post(api_url, json=results) as response:
            if response.status != 200:
                print(f"Failed to post results: {response.status}")
                return False
            return True

def get_mock_image_data():
    return [
        {'id': '1', 'url': 'https://m.media-amazon.com/images/I/61IahNYDi0L.jpg'}, #True
        {'id': '2', 'url': 'https://m.media-amazon.com/images/I/41j0X3rxIgL.jpg'}, #False
        {'id': '3', 'url': 'https://static.mercdn.net/item/detail/orig/photos/m80964885046_1.jpg'},
        {'id': '4', 'url': 'https://static.mercdn.net/item/detail/orig/photos/m19309783385_1.jpg'},
        {'id': '5', 'url': 'https://m.media-amazon.com/images/I/61IahNYDi0L.jpg'},
    ]

def main():
    try:
        start_time = time.time()
        
        # APIからデータを取得
        image_data = asyncio.run(fetch_image_data_from_api(limit=10, sellerId=''))
        

        if len(image_data) == 0:
            print("No images to process")
            return

        # 絶対パスを使用するように変更
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logos_dir = os.path.join(os.path.dirname(current_dir), 'data', 'logos')

        if not os.path.exists(logos_dir):
            print(f"Error: Logos directory '{logos_dir}' does not exist")
            return

        # 非同期処理の実行
        results = asyncio.run(process_multiple_images(image_data, logos_dir))
        
        # 結果の取得
        json_results = display_results(results, show_images=False)
        
        # 検出数の集計
        detected_count = sum(1 for result in json_results if result['brands'] is not None)
        
        # 結果をAPIにPOST
        success = asyncio.run(post_results_to_api(json_results))
        
        # 処理時間と検出数の出力
        elapsed_time = time.time() - start_time
        print(f"Detected: {detected_count}/{len(json_results)} logos in {round(elapsed_time, 0)} seconds")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()