from flask_script import Manager, Server
from main import app

# 然后把你的app传给Manager对象，以初始化Flask Script:
manager = Manager(app)


# 现在我们来添加一些命令。这里运行的服务器跟通过main.py运行的普通开发服务器是一样的。make_shell_context函
# 数会创建一个Python命令行，并且在应用上下文中执行。返回的字典告诉Flask Script在打开命令行时进行一些默认
# 的导入工作。
manager.add_command("server", Server())

@manager.shell
def make_shell_context():
    return dict(app=app)

# 通过manage.py运行命令行在将来会十分必要，因为一些Flask扩展只有在Flask应用对象被创建之后才会被初始化。直接
# 运行默认的Python命令行会令这些扩展返回错误。


# 然后，在文件结尾添加如下代码，这是Python的标准方式，用来限制仅在用户直接运行文件的时候，才执行上面的python3 代码：
if __name__ == "__main__":
    manager.run()

