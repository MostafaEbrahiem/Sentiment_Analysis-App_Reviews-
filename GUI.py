from tkinter import *
from operation import test_custom_data


form=Tk()
form.geometry("500x500")
form.title("Sentement_Analysis")

sentence=Label(text="Enter a sentence")
sentence.place(x=15,y=30)

result=Label(text="")
result.place(x=240,y=150)


sentence_text=Text(height="1",width="25")
sentence_text.place(x=210,y=33)

ok_butt=Button(text="   check   ",command=test_custom_data)
ok_butt.place(x=230,y=400)

form.mainloop()