library(tidyverse)

pdat <- read_csv("scores_trained_on_all.csv") %>% 
  select(-X1) %>%
  mutate(measured=ifelse((real%%1)==0,T,F))

g <- pdat%>%ggplot()+theme_minimal()+
  aes(x=real,y=pred)+
  stat_smooth(method="lm",formula="y~x",col="black",linetype="dashed")+
  geom_violin(width=0.1,aes(group=factor(real),fill=measured))+
  scale_y_continuous(limits=c(0,6))
g

ggsave("predictions_vs_real.jpg",g)

g<- pdat%>%group_by(real)%>%select(-measured)%>%
  summarize_all(mean)%>%gather(Variable,Value,-real,-pred)%>%
  ggplot()+aes(x=Variable,y=Value)+theme_minimal()+
  geom_bar(stat="identity")+facet_wrap(~real)+
  theme(axis.text.x=element_text(angle=90))
g

ggsave("importance_of_variables_within_classes.jpg",g,width=14)

g<- pdat%>%select(-measured)%>%
  summarize_all(mean)%>%gather(Variable,Value,-real,-pred)%>%
  ggplot()+aes(x=Variable,y=Value)+theme_minimal()+
  geom_bar(stat="identity")+
  theme(axis.text.x=element_text(angle=90))
g

ggsave("importance_of_variables.jpg",g)

flow_dat <- read_csv("data/fl_cn_training_data.csv",
    col_types=str_c(rep("d",15),collapse=""))%>%
  mutate(standards=ifelse((SL_RD_copy_estimate%%1)==0,T,F))

g <- flow_dat %>% 
  ggplot()+theme_minimal()+
  aes(x=SL_RD_copy_estimate)+
  geom_violin(aes(group=factor(SL_RD_copy_estimate),
      fill=standards),width=0.1)+
  stat_smooth(method="lm",formula="y~x",col="black",linetype="dashed")

ggsave("FL1area.jpg",g+aes(y=log10(FL1.A)))
ggsave("FL1height.jpg",g+aes(y=log10(FL1.H)))
ggsave("FL2height.jpg",g+aes(y=log10(FL2.H)))
ggsave("FL2HoverFSCa.jpg",g+aes(y=log10(FL2.H/FSC.A)))
ggsave("FL2HoverSSCA.jpg",g+aes(y=log10(FL2.H/SSC.A)))
ggsave("FL2HoverFSCASSCA.jpg",g+aes(y=log10(FL2.H/(FSC.A+SSC.A))))

cor_table <- flow_dat%>%select(-Time,-standards)%>%cor(method="spearman")
cor_table['SL_RD_copy_estimate',]

as.tibble(cor_table)%>%mutate(Var1=rownames(cor_table))%>%
  gather(Var2,Value,-Var1)%>%
  ggplot()+aes(x=Var1,y=Var2,size=exp(Value),col=exp(Value))+
  geom_point()+theme_minimal() 


