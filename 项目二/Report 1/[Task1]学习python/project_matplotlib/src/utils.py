import  os

class utils:
    """
    自定义工具类
    """
    
    @staticmethod
    def getPath():
        """返回项目根目录"""
        return os.path.abspath('..')
