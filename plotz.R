library(tidyverse)

pdat <- read_csv("scores_by_predictions.csv") %>% select(-X1) 

pdat %>%
  gather(Variable,Value,-real,-pred)%>%
  group_by(Variable,real,pred)%>%
  summarize(Value=mean(Value))%>%
  ggplot()+
  aes(x=pred,y=Value,fill=Variable)+
  geom_bar(stat="identity",position="dodge")+
  facet_wrap(~real,scales="free")




#pdat <- list.files(path="./",pattern="scores.*\\.csv") %>% 
#  tibble(Filename=.) %>% 
#  mutate(RawFile=map(Filename,read_csv,col_names=F),
#    which_score=str_match(Filename,"\\d")) %>%
#  unnest %>%
#  gather(Variable,Value,starts_with("X")) %>%
#  select(-Filename)
#
#g <- pdat %>%
#  ggplot()+
#  aes(x=Value)+geom_histogram(bins=100)+
##  scale_x_continuous(limits=c(-100,100))+
#  scale_y_log10()+
#  theme(axis.text.x=element_text(angle=90))+
#  facet_grid(Variable~which_score)
#g
#
#ggsave("scores_by_output_class_and_input.jpg",g)
#
#g <- pdat%>%group_by(which_score,Variable)%>%
#  summarize(Mean=mean(Value),Var=var(Value)) %>% 
#  gather(NewVariable,Value,Mean,Var) %>% 
#  ggplot()+aes(x=Variable,y=Value)+geom_bar(stat="identity")+
#  coord_flip()+
#  facet_grid(which_score~NewVariable,scales="free")
#g
#
#ggsave("summary_scores_by_output_class.jpg",g)
