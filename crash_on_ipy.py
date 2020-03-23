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
    print sys._getframe().f_code.co_filename  # ��ǰ�ļ���
    print sys._getframe(0).f_code.co_name  # ��ǰ������
    print sys._getframe(1).f_code.co_name��# ���øú����ĺ��������֣����û�б����ã��򷵻�module
    print sys._getframe().f_lineno # ��ǰ�к�