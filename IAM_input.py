import os
import numpy as np
from PIL import Image
import cv2
class IAM_input:
    def __init__(self,partition="train"):
        self.basedir="/home/mcarbonell/Documents/DATASETS/IAM/lines"
        self.partition=partition
        self.symbsList="|abcdefghijklmnopqrstuvwxyz"
        self.symbsDict = dict([(x[1], x[0]) for x in enumerate(self.symbsList)])
        self.index=0
        self.create_line_iterator()
        self.line_id=self.line_iterator.next()
        self.index_in_epoch=0
        self.batch_size=1
        self.im_height=64.

    def char_to_id(self,sentence):
        ids=[]
        for letter in sentence:
            if self.symbsDict.get(letter)!=0 and not self.symbsDict.get(letter):
                continue

            else:
                ids.append(self.symbsDict.get(letter))
        if len(ids)==0:
            ids.append(len(self.symbsDict))
        ids=np.stack(ids)

        #print sentence,ids,type(ids)
        return ids

    def create_line_iterator(self):
        f = open(os.path.join(self.basedir, 'lines_'+self.partition+'.txt'))
        lines = f.readlines()
        line_ids=[]
        self.total_examples=0
        self.transcriptions={}
        for line in lines:
            if line[0]=="#":
                continue
            cols=line.split(" ")
            line_id=cols[0]
            line_ids.append(line_id)
            sentence=cols[-1]
            sentence=sentence.strip(',.').lower()
            self.transcriptions[line_id]=self.char_to_id(sentence)
            self.total_examples=self.total_examples+1

        self.line_iterator = iter(line_ids)


    def get_image(self,line_id):
        image_path=os.path.join(self.basedir,line_id.split("-")[0])
        image_path=os.path.join(image_path,"-".join(line_id.split("-")[0:2]))
        image_path = os.path.join(image_path, line_id+'.png')

        im2 = cv2.imread(image_path)
        size=im2.shape

        im2=cv2.resize(im2,None,fx=self.im_height /size[0],fy=self.im_height/size[0], interpolation = cv2.INTER_CUBIC)

        im2=255-im2

        im2 = im2[:, :, 0]

        def binarize(im,threshold=120):
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    if im[i,j]<threshold or not im2[i,j]:
                        im[i,j]=0
                    else:
                        im[i, j] = 255
            return im

        im2=binarize(im2)

        #im2=im2/255.
        return im2[:,:,np.newaxis]

    def get_batch(self):
        X=[]
        Y=[]
        for i in range(self.batch_size):
            try:
                self.line_id=self.line_iterator.next()

            except StopIteration:
                self.create_line_iterator()
                self.index_in_epoch = 0
                self.line_id = self.line_iterator.next()

            X.append(self.get_image(self.line_id))
            Y.append(self.transcriptions[self.line_id])
            self.index_in_epoch += 1
        X=np.stack(X)

        return X,Y
        

def main():
    iam=IAM_input()
    X,Y=iam.get_batch()
    for i in range(len(X)):
        im=Image.fromarray(X[i,:,:,0],mode="L")

        im.show()
        print(Y)
        raw_input()
if __name__=="__main__":
    main()