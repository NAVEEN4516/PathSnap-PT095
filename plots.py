"""Create portable PNG charts from the recorded CSVs, using Pillow."""
import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from prepare import ROOT

def chart(filename,title,labels,series,ceiling=1.0):
    image=Image.new("RGB",(1100,650),"#f5f7f4")
    d=ImageDraw.Draw(image)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default(size=18)
    d.text((35,25),title,fill="#173e35",font=font)
    left,top,bottom=95,110,540
    colors=["#187969","#b85235","#5079a2"]
    for i in range(6):
        y=bottom-i*(bottom-top)/5
        d.line((left,y,1060,y),fill="#d9dfd9")
        d.text((20,y-8),f"{ceiling*i/5:.2f}",fill="#294b40",font=font)
    width=950/len(labels)
    for j,(name,values) in enumerate(series.items()):
        d.rectangle((120+j*300,65,135+j*300,80),fill=colors[j])
        d.text((145+j*300,61),name,fill="#294b40",font=font)
        for i,value in enumerate(values):
            x=left+i*width+8+j*width/(len(series)+1)
            bar=width/(len(series)+1)-5
            y=bottom-value/ceiling*(bottom-top)
            d.rectangle((x,y,x+bar,bottom),fill=colors[j])
    for i,label in enumerate(labels):
        d.text((left+i*width+4,bottom+15),label,fill="#294b40",font=font)
    dest=ROOT/"results/plots";dest.mkdir(exist_ok=True)
    image.save(dest/filename)

def main():
    with (ROOT/"results/comparison.csv").open() as f:rows=list(csv.DictReader(f))
    labels=[r["model"]+("*" if r["stage"]!="baseline" else "") for r in rows]
    chart("f1.png","Macro F1 by model (* Bayesian search winner)",labels,
          {"Group CV":[float(r["cv_f1"]) for r in rows],"Student test":[float(r["macro_f1"]) for r in rows]})
    chart("accuracy.png","Student-test accuracy and immediate road continuity",labels,
          {"Top-1":[float(r["top1"]) for r in rows],"Continuity":[float(r["continuity"]) for r in rows]})
    chart("log_loss.png","Candidate binary log loss",labels,
          {"Log loss":[float(r["log_loss"]) for r in rows]},.3)
if __name__=="__main__":main()
