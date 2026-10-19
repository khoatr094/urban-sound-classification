from lxml import etree
r=etree.parse('outputs/figures/14_fps_stability_accuracy_comprehensive.svg');ns={'s':'http://www.w3.org/2000/svg'}
a=r.xpath('//s:g[starts-with(@id,"axes_")]',namespaces=ns)[3]
for g in a.xpath('.//s:g[starts-with(@id,"text_")]',namespaces=ns):
 c=g.xpath('comment()');t=c[0].text if c else ''
 if any(x in t for x in ['CNN','AST','MB']):print(t,[(ch.tag,ch.get('transform')) for ch in g if isinstance(ch.tag,str)])


