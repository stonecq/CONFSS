import json
import logging
def save_as_json(data, filename):
    """保存数据为JSON文件（带错误处理）"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"数据已保存至 {filename}")
    except (IOError, TypeError) as e:
        logging.error(f"保存JSON文件失败: {e}")