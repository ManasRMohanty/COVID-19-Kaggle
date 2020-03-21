from CNER_BertUtility import *
from pylab import *
import pandas as pd
import copy

def is_abbrev(text,abbrev):
    text = text.lower()
    abbrev = abbrev.lower()
    remaining_text = text
    for entry in abbrev:
        pos = remaining_text.find(entry)
        if(pos==-1):
            return False
        else:
            remaining_text = remaining_text[pos:]
    
    return True
    
def resolve_continuity(text,words,ignore_letter_list=[]): #"\u03B2","\u03B1","\u03B3","\u03B4","\u03B8","\u03BC","\u03C3"
    list_of_positions = []
    word_list_cont = []
    first_entry = True
    prev_entry = None
    
    
    for index, row in words.iterrows():
        if(index>0 and row['word'].lower() in ['acid', 'alcohol']):
            prev_row = words.iloc[index-1]
            if(prev_row['chemical_predict']==1 and row['chemical_predict']==1):
                continue
            elif(prev_row['chemical_predict']==1 or row['chemical_predict']==1):
                if(prev_row['pos_token'] in ['JJ','JJR','JJS','NN','NNS','NNP','NNPS','MD']):
                    words.at[index-1,'chemical_predict'] = words.at[index,'chemical_predict'] = 1
                    
    
    
    words_dict = list(words.T.to_dict().values())
    
    index = -1
    
    for entry in words_dict:
        index = index + 1
        entry["ignore_entry"] = False
        del entry["keyword_vector"]
        
        if(entry['chemical_predict']==1):
            if(entry['word'] in ['and', 'as','mg','of','in']):
                entry['chemical_predict'] = 0      
        '''
        else:
            if(entry["word"] in periodic_element_list):
                entry['chemical_predict'] = 1
            if(re.match("[NCH]-.*",entry["word"])):
                entry["word"] = text[entry["begin_pos"]]
                entry["end_pos"] = entry["begin_pos"] + 1
                entry['chemical_predict'] = 1
        '''
        
        if(not first_entry):
            in_between_text = text[prev_entry["end_pos"]:entry["begin_pos"]]
            for letter in ignore_letter_list:
                in_between_text = re.sub(letter, ' ', in_between_text)
            if(prev_entry['chemical_predict']==1):
                if(entry['chemical_predict'] == 1):
                    if(len(in_between_text.strip())==0):
                        if(entry['word'] in ['and', 'as','mg'] ):
                            entry['chemical_predict'] = 0
                            word_list_cont.append(prev_entry)
                            prev_entry = entry
                        else:
                            prev_entry["end_pos"] = entry["end_pos"]
                            prev_entry["word"] = text[prev_entry["begin_pos"]:prev_entry["end_pos"]]
                    else:
                        word_list_cont.append(prev_entry)
                        prev_entry = entry
                else:
                    if(len(in_between_text)==0 and re.match("[A-Za-z0-9]",entry["word"][0])):
                        prev_entry["end_pos"] = entry["end_pos"]
                        prev_entry["word"] = text[prev_entry["begin_pos"]:prev_entry["end_pos"]]
                    else:
                        word_list_cont.append(prev_entry)
                    
                        if(entry["word"]=="("):
                            next_entry = words_dict[index+1]
                            next_to_next_entry = words_dict[index+2]
                            if(next_entry['chemical_predict']!=1 and next_to_next_entry["word"]==")"):
                                if(is_abbrev(prev_entry["word"],next_entry["word"])):
                                    next_entry['chemical_predict'] = 1
                        prev_entry = entry
            else:
                prev_entry = entry
        else:
            prev_entry = entry
            first_entry =False
            
    if(prev_entry is not None and prev_entry['chemical_predict']==1):
        word_list_cont.append(prev_entry)
    
    new_add_list = []
    for entry in word_list_cont:
        if(entry["ignore_entry"]):
            continue
        if(len(entry["word"])==1):
            continue
            
        if(entry["word"].lower() not in["and","as","in","mg"]):
            begin_words = [i for i in range(len(text)) if text.lower().startswith(entry["word"].lower(), i)]
        else:
            begin_words = [i for i in range(len(text)) if text.startswith(entry["word"], i)]
            
        end_words = [i+len(entry["word"]) for i in begin_words]
        
        index = -1
        for begin_position in begin_words:
            index = index + 1
            if(begin_position==entry["begin_pos"]):
                continue
            end_position = end_words[index]
            if re.match('[a-zA-Z0-9]',text[end_position]):
                continue
            if(begin_position > 0 and re.match('[a-zA-Z0-9]',text[begin_position-1])):
                continue
            found = False
            
            for entry_in in word_list_cont:
                if(entry_in["begin_pos"]==entry["begin_pos"]):
                    continue
                elif(entry_in["begin_pos"] == begin_position and entry_in["end_pos"] == end_position):
                    entry_in.update({"ignore_entry":True})
                    found = True
                    break
                elif(entry_in["begin_pos"] <= begin_position and entry_in["end_pos"] >= end_position):
                    found = True
                    break
                elif(entry_in["begin_pos"] <= begin_position and entry_in["end_pos"] > begin_position and entry_in["end_pos"] < end_position):
                    entry_in.update({"end_pos":end_position})
                    entry_in.update({"word":text[entry_in["begin_pos"]:entry_in["end_pos"]]})
                    found = True
                    break
                elif(entry_in["begin_pos"] > begin_position and entry_in["begin_pos"] < end_position and entry_in["end_pos"] >= end_position):
                    entry_in.update({"begin_pos":begin_position})
                    entry_in.update({"word":text[entry_in["begin_pos"]:entry_in["end_pos"]]})
                    found = True
                    break
            
            if(not found):
                entry_new = copy.deepcopy(entry)
                entry_new["begin_pos"] = begin_position
                entry_new["end_pos"] = end_position
                new_add_list.append(entry_new)
                
    if(len(new_add_list)>0):
        word_list_cont.extend(new_add_list)
    
    word_list_cont = sorted(word_list_cont, key = lambda i: i['begin_pos'])
    
    return pd.DataFrame(word_list_cont)

def get_chemicals_from_text(text,padding_length=1):
    mod_text = text.replace("/"," ")
    
    word_list_df = pd.DataFrame(process_string_finetune(mod_text,padding_length))

    clf = pickle.load(open("/kaggle/input/covid19/chemical_model_complete_base.pkl","rb"))
    
    X = np.vstack(list(word_list_df["keyword_vector"]))
    
    word_list_df['chemical_predict'] = clf.predict(X)

    word_list_df = resolve_continuity(text,word_list_df)
    
    return word_list_df

def process_text(text):
    if(text.isdecimal()):
        return process_text_from_pubmed(text)
    
    word_list_df = get_chemicals_from_text(text,0)
    
    chemical_cmap = cm.get_cmap('YlOrBr', 1000)
    last_position = 0
    processed_text = ""

    for index,entry in word_list_df.iterrows():
        processed_text = processed_text + text[last_position:entry["begin_pos"]]
        
        if(entry["chemical_predict"]==1):
            color_code = matplotlib.colors.rgb2hex(chemical_cmap(500)[:3])
            processed_text = processed_text + "<span style=\"background-color:"+color_code+";\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
        
        last_position = entry["end_pos"]
    

    processed_text = processed_text + text[last_position:]

    processed_text.replace('\n', '<br>').replace('\r', '<br>')
    return "<br />".join(processed_text.split("\n"))