from tkinter import * 
from tkinter.messagebox import askyesno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import ImageOps
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from PIL import ImageGrab

#load the model from disk
NN_model = load_model('best_mnist_aug.hdf5')


class GUI():

    def __init__(self):

        if(os.path.isfile('dataset.pkl')):
            self.df = pd.read_pickle('dataset.pkl')
        else:
            self.df = pd.DataFrame(columns=["image_data", "label"])

        self.index_pointer = len(self.df.index)-1 #last index of df
        self.last_index = self.index_pointer

        self.root = Tk()
        #self.root.resizable(width=True, height=True)
        self.root.geometry("840x730")
        #self.root.state("zoomed")
        self.root.configure(bg="#ffa5a4")

        self.canvas_write = Canvas(self.root, bg='white', width=400, height=400)
        self.canvas_write.place(x = 10, y = 10)

        self.fig , self.ax = plt.subplots(figsize=(4,4)) 
        self.canvas_read = FigureCanvasTkAgg(self.fig, master = self.root)
        self.canvas_read.get_tk_widget().place(x=430, y=10)
        plt.imshow(self.df["image_data"][self.index_pointer])
        


        self.before_button = Button(self.root, text='<---', height= 1, width=10, command=self.show_previous)
        self.before_button.place(x=430, y=430)

        self.index_label = Label(self.root, text="Index:", width=14, anchor="w")
        self.index_label.place(x=520, y=430)
        self.index_label.config(text = "Index: "+str(self.index_pointer))

        self.label_label = Label(self.root, text="Given Label:", width=14, anchor="w")
        self.label_label.place(x=635, y=430)
        self.label_label.config(text = "Given Label: "+str(self.df["label"][self.index_pointer]))
        
        self.after_button = Button(self.root, text='--->', height= 1, width=10, command=self.show_next)
        self.after_button.place(x=755, y=430)

        self.add_label = Label(self.root, text="New Label:",width=10 )
        self.add_label.place(x=700, y=470)

        self.add_entry = Entry (self.root , width=3, justify='center')
        self.add_entry.place(x=780, y=471)
        self.add_entry.insert(0,'0')

        self.append_button = Button(self.root, text='Append', height= 2, width=12, command=lambda:self.add(True), bg='#DAF7A6')
        self.append_button.place(x=556, y=470)
        
        self.insert_button = Button(self.root, text='Insert', height= 2, width=12, command=lambda:self.add(False))
        self.insert_button.place(x=430, y=470)

        self.delete_button = Button(self.root, text='Delete at Current Index', height= 2, width=30, command=self.delete)
        self.delete_button.place(x=430, y=520)

        self.save_all_button = Button(self.root, text='Save All Data', height= 2, width=30, command=self.save_all, bg='#DAF7A6')
        self.save_all_button.place(x=430, y=570)

        self.predict_button = Button(self.root, text='Predict', command=self.predict, height= 3, width=20, bg='#DAF7A6')
        self.predict_button.place(x=30,y=440)

        self.clear_screen_button = Button(self.root, text='Clear Screen', command=self.clear_screen, height= 3, width=20, bg='#FFD1DC')
        self.clear_screen_button.place(x=230,y=440)

        self.NN_label = Label(self.root, text="Predicted Value:   ", font=('Helvetica bold', 32), width=16, bg="#FFE5B4")
        self.NN_label.place(x=10,y=510)

        self.console_label = Label(self.root, text="Console:")
        self.console_label.place(x=10,y=590)

        self.console = Text(self.root, height=6)
        self.console.place(x=10,y=620)

        self.keypad = Frame(master= self.root)
        self.keypad.place(x=670, y=500)

        self.num1 = Button(self.keypad, text='1', height= 3, width=6,command= lambda : self.enter_label('1'))
        self.num1.grid(row=0, column=0)

        self.num2 = Button(self.keypad, text='2', height= 3, width=6, command= lambda : self.enter_label('2'))
        self.num2.grid(row=0, column=1)

        self.num3 = Button(self.keypad, text='3', height= 3, width=6, command= lambda : self.enter_label('3'))
        self.num3.grid(row=0, column=2)

        self.num4 = Button(self.keypad, text='4', height= 3, width=6, command= lambda : self.enter_label('4'))
        self.num4.grid(row=1, column=0)

        self.num5 = Button(self.keypad, text='5', height= 3, width=6, command= lambda : self.enter_label('5'))
        self.num5.grid(row=1, column=1)

        self.num6 = Button(self.keypad, text='6', height= 3, width=6, command= lambda : self.enter_label('6'))
        self.num6.grid(row=1, column=2)

        self.num7 = Button(self.keypad, text='7', height= 3, width=6, command= lambda : self.enter_label('7'))
        self.num7.grid(row=2, column=0)

        self.num8 = Button(self.keypad, text='8', height= 3, width=6, command= lambda : self.enter_label('8'))
        self.num8.grid(row=2, column=1)

        self.num9 = Button(self.keypad, text='9', height= 3, width=6, command= lambda : self.enter_label('9'))
        self.num9.grid(row=2, column=2)

        self.num0 = Button(self.keypad, text='0', height= 3, width=6, command= lambda : self.enter_label('0'))
        self.num0.grid(row=3, column=0)

        self.numx = Button(self.keypad, text='X', height= 3, width=14, command= lambda : self.enter_label(''))
        self.numx.grid(row=3, column=1, columnspan=2)


        def on_closing():
            self.root.quit()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 50
        self.canvas_write.bind('<B1-Motion>', self.paint)
        self.canvas_write.bind('<ButtonRelease-1>', self.reset)

    def plot(self):
        self.fig.clear()
        self.index_label.config(text = "Index: "+str(self.index_pointer))
        self.label_label.config(text = "Given Label: "+str(self.df["label"][self.index_pointer]))
        plt.imshow(self.df["image_data"][self.index_pointer])
        self.canvas_read.draw()

    def show_previous(self):
        if self.index_pointer>0:
            self.index_pointer-=1
            self.plot()
        else:
            self.console_print("First Element Reached")


    def show_next(self):
        if self.index_pointer<self.last_index:
            self.index_pointer+=1
            self.plot()
        else:
            self.console_print("Last Element Reached")

    def get_image(self, widget):
        x=self.root.winfo_rootx()+widget.winfo_x()
        y=self.root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        return ImageGrab.grab().crop((x,y,x1,y1))

    def add(self, mode:bool):
        """For Appending and Inserting value to the dataframe

        Args:
            mode (bool): True For Appending, False For inserting at current index
        """

        img = self.get_image(self.canvas_write)

        img = ImageOps.invert(img)
        img = img.convert("L") #convert to gray scale
        img = img.resize((32,32))
        arr = np.array(img)

        if(not np.any(arr)):#if image is empty:
            self.console_print("Error: No Image drawn")
            return

        label_data = self.add_entry.get()
        if(label_data not in ['0','1','2','3','4','5','6','7','8','9']):
            self.console_print("Error: Entered values must be in range 0-9")
            return
        else:
            label_data = int(label_data)
         
        #inserting data at current index   
        
        if(mode):
            self.df.loc[len(self.df.index)] = [arr, label_data]
            self.last_index = len(self.df.index)-1
            self.index_pointer+=1
            self.clear_screen()
            self.plot()
            self.console_print("Added: "+str(label_data))
            self.console_print("Total no. of values in dataset:"+ str(self.last_index))
            
            label_data+=1
            if(label_data==10):
                label_data = 0
            self.add_entry.delete(0, END)
            self.add_entry.insert(0,label_data)
             
        else:
            line = pd.DataFrame(columns=["image_data", "label"])
            line.loc[0] = [arr, label_data] 
            self.df = pd.concat([self.df.iloc[:self.index_pointer], line, self.df.iloc[self.index_pointer:]]).reset_index(drop=True)
            self.last_index = len(self.df.index)-1
            self.clear_screen()
            self.plot()
            self.console_print("Added:"+str(label_data))
            self.console_print("Total no. of values in dataset:"+ str(self.last_index))
        

            
    def delete(self):
        self.df.drop(self.index_pointer,inplace=True)
        
        if self.index_pointer == self.last_index:
            self.index_pointer-=1

        self.last_index = len(self.df.index)-1
        self.df.reset_index(inplace=True,drop=True)
        self.add_entry.delete(0, END)
        self.clear_screen()
        self.plot()
        
    def save_all(self):
        answer = askyesno(title='confirmation',
                    message='Are you sure you wish to save all changes?')
        
        if answer:
            self.df.to_pickle('dataset.pkl')
            self.console_print("All Changes Saved")


    def enter_label(self, val):
        self.add_entry.delete(0, END)
        self.add_entry.insert(0,val)


    def predict(self):

        img = self.get_image(self.canvas_write)

        img = ImageOps.invert(img)
        img = img.convert("L") #convert to gray scale
        img = img.resize((32,32))
        arr = np.array(img) / 255.0
        
        result = NN_model.predict(np.array([arr]), verbose = 0)
        self.NN_label.config(text = "Predicted Value: "+str(np.argmax(result)))
        self.console_print("Predicted Value: "+str(np.argmax(result)))

    def clear_screen(self):
        self.canvas_write.delete("all")
        self.NN_label.config(text = "Predicted Value:   ")

    def paint(self, event):
        self.line_width = 30
        paint_color = 'black'
        if self.old_x and self.old_y:
            self.canvas_write.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def console_print(self, text):
        self.console.insert(END, ">>>"+text+"\n")
        self.console.see(END)


if __name__ == '__main__':
    GUI()
    