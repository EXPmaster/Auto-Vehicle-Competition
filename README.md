#### README

1. 目录树

   ```
   |-- AVC
   	  |-- .gitignore
   	  |-- labeled_data_backup.zip
   	  |-- labeled_data_backup
   	  |		|-- x.jpg
   	  |		|-- x.xml
   	  |-- Utils.py
   	  |-- Dataloader.py
   	  |-- train.py
   	  |-- model.py
   ```

   

2. 代码接口

   Utils.py

   ```python
   class Config:
     pass
   
   class Metric:
     def __init__():
       pass
     def update():
       pass
     def reset():
       pass
   ```

   model.py

   ```python
   class xxx:
     def __init__():
       pass
     def forward():
       pass
     # ...
   ```

   train.py

   ```python
   def train():
     pass
   
   def val():
     pass
   
   def main():
     pass
   
   if __name__ == '__main__':
     main()
   ```

   （optional）test.py

   ```python
   def test()：
   	pass
   
   if __name__ == '__main__':
     test()
   ```

   