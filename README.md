<font style="color:rgb(31, 35, 40);">This work is being submitted to the WSDM, and it will be gradually improved based on the feedback.</font>

### <font style="color:rgb(31, 35, 40);">Raw_Dataset</font>
| **<font style="color:rgb(31, 35, 40);">Dataset</font>** | **<font style="color:rgb(31, 35, 40);">Resource</font>** |
| --- | --- |
| Weibo-comp | [https://www.datafountain.cn/competitions/422](https://www.datafountain.cn/competitions/422) |
| <font style="color:rgb(31, 35, 40);">RumourEval-19</font> | [https://aclanthology.org/S19-2147/](https://aclanthology.org/S19-2147/) |


<font style="color:rgb(31, 35, 40);">You can customize the dataset to your needs. For this work, we processed the datasets and provided the dataset file, you can download the dataset </font><font style="color:rgba(0, 0, 0, 0.85);"> in the folder "data_with_emotion".</font>

### <font style="color:rgba(0, 0, 0, 0.85);">Resource</font>
you can download the embedding files as the following table:

| **embedding file** | **link** |
| --- | --- |
| glove.6B.100d | [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) |
| sgns.weibo.word | [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) |


<font style="color:rgb(31, 35, 40);">After downloading the </font>**embedding file**<font style="color:rgb(31, 35, 40);">, please ensure to place it in the appropriate folder as demonstrated below:</font>

```bash
-resource
   --embedding
     ---glove.6B.100d.txt
     ---sgns.weibo.word
       ----sgns.weibo.word
```

### <font style="color:rgba(0, 0, 0, 0.85);">Code</font>
**<font style="color:rgb(31, 35, 40);">Requirements</font>**<font style="color:rgb(31, 35, 40);">:</font>

```bash
Python=3.8.20
torch=1.5.1
torchvision=0.6.1_cu102
nltk=3.9.1
numpy=1.24.3
```



you can train model by running train.py

