import sys

class ExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode='Plain',
                 color_scheme='Linux', call_pdb=1)
        return self.instance(*args, **kwargs)

sys.excepthook = ExceptionHook()

def get_cur_info():
    print sys._getframe().f_code.co_filename  # 当前文件名
    print sys._getframe(0).f_code.co_name  # 当前函数名
    print sys._getframe(1).f_code.co_name　# 调用该函数的函数的名字，如果没有被调用，则返回module
    print sys._getframe().f_lineno # 当前行号