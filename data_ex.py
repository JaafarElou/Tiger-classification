from bing_image_downloader import downloader

downloader.download("owl", limit=5, output_dir=r"tiger_data\train", adult_filter_off=False)