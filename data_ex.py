from bing_image_downloader import downloader

downloader.download("deer", limit=5, output_dir=r"tiger_data\train\not_tiger", adult_filter_off=False)
