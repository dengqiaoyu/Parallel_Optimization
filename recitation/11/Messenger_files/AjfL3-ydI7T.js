if (self.CavalryLogger) { CavalryLogger.start_js(["\/MUMB"]); }

__d('MessengerContextualActions.react',['cx','ContextualLayerUpdateOnScroll','LayerHideOnBlur','Link.react','React','ReactAbstractContextualDialog','ReactLayer'],(function a(b,c,d,e,f,g,h){'use strict';var i,j,k,l,m=c('React').PropTypes,n=c('ReactLayer').createClass(c('ReactAbstractContextualDialog').createSpec({displayName:'MessengerContextualActionsDialog',theme:{wrapperClassName:"_hw2",arrowDimensions:{offset:12,length:12}}}));i=babelHelpers.inherits(o,c('React').Component);j=i&&i.prototype;o.prototype.render=function(){var q=this.props,r=q.children,s=babelHelpers.objectWithoutProperties(q,['children']);return c('React').createElement(n,babelHelpers['extends']({alignment:'center',behaviors:{ContextualLayerUpdateOnScroll:c('ContextualLayerUpdateOnScroll'),LayerHideOnBlur:c('LayerHideOnBlur')}},s,{onToggle:this.props.onToggle}),c('React').createElement('ul',{className:"_hw3"},r));};function o(){i.apply(this,arguments);}o.propTypes={onToggle:m.func};o.Item=p;k=babelHelpers.inherits(p,c('React').Component);l=k&&k.prototype;p.prototype.render=function(){return c('React').createElement('li',{className:"_hw4"},c('React').createElement(c('Link.react'),{className:"_hw5",onClick:this.props.onClick},this.props.children));};function p(){k.apply(this,arguments);}p.propTypes={onClick:m.func};f.exports=o;}),null);
__d("XMessengerDotComSaveController",["XController"],(function a(b,c,d,e,f,g){f.exports=c("XController").create("\/save\/message\/",{});}),null);
__d('messengerDeleteMessageDialogReact',['fbt','bs_curry','reactRe','MercuryConfig','messengerDialogReact','messengerDialogBodyReact'],(function a(b,c,d,e,f,g,h){'use strict';var i=+c('MercuryConfig').MessengerGlobalDeleteGK,j=c('reactRe').Component[9],k=j[0],l=j[1],m=j[2],n=j[3],o=j[4],p=j[5],q=j[6],r="MessengerDeleteMessageDialogReact";function s(ea,fa){var ga=ea[1];c('bs_curry')._1(ga[3],false);c('bs_curry')._1(ga[2],0);return 0;}function t(ea,fa){c('bs_curry')._1(ea[1][1],0);return 0;}function u(ea){if(i!==0){if(ea!==0){return h._("\u5220\u9664\u6d88\u606f\uff1f");}else return h._("\u9690\u85cf\u6d88\u606f\uff1f");}else return h._("\u5220\u9664\u6d88\u606f");}function v(ea){if(i!==0){if(ea!==0){return h._("\u6d88\u606f\u4e00\u65e6\u5220\u9664\uff0c\u6240\u6709\u804a\u5929\u5bf9\u8c61\u90fd\u4e0d\u4f1a\u518d\u770b\u5230\u6d88\u606f\uff0c\u64cd\u4f5c\u65e0\u6cd5\u64a4\u9500\u3002");}else return h._("\u6d88\u606f\u4e00\u65e6\u9690\u85cf\uff0c\u4f60\u5c06\u6ca1\u6cd5\u518d\u627e\u56de\u3002\u5bf9\u65b9\u4ecd\u7136\u53ef\u4ee5\u770b\u5230\u8fd9\u6761\u6d88\u606f\u3002");}else return h._("\u4f60\u786e\u5b9a\u8981\u5220\u9664\u8fd9\u6761\u4fe1\u606f\uff1f");}function w(ea){if(i!==0){if(ea!==0){return h._("\u5220\u9664");}else return h._("\u9690\u85cf");}else return h._("\u5220\u9664");}function x(ea){var fa=ea[2],ga=ea[1];return c('bs_curry').app(c('messengerDialogReact').Dialog[0],[0,ga[3],0,0,0,0,0,[c('bs_curry')._6(c('messengerDialogReact').Header[0],0,0,[u(ga[0]),0],0,0,0),[c('bs_curry')._3(c('messengerDialogBodyReact').createElement(0,[v(ga[0]),0]),0,0,0),[c('bs_curry')._6(c('messengerDialogReact').Footer[0],0,0,[c('bs_curry')._5(c('messengerDialogReact').CancelButton[0],[c('bs_curry')._1(fa,t)],0,0,0,0),[c('bs_curry').app(c('messengerDialogReact').Button[0],[0,[w(ga[0])],0,["danger"],[c('bs_curry')._1(fa,s)],[0],0,0,0,0,0,0]),0]],0,0,0),0]]],0,0,0]);}var y=[function(ea){return [+ea.isFromViewer,ea.onCancel,ea.onDelete,ea.onToggle];}],z=[k,l,m,n,o,p,q,r,s,t,u,v,w,x,y],aa=c('reactRe').CreateComponent([r,p,q,k,n,l,m,o,y,x]),ba=aa[1],ca=c('bs_curry').__1(ba),da=aa[0];g.globalDelete=i;g.MessengerDeleteMessageDialog=z;g.comp=da;g.wrapProps=ba;g.createElement=ca;}),null);
__d("messengerSaveMessageDialogReact",["fbt","URI","bs_curry","reactRe","linkReact","messengerDialogReact","messengerDialogBodyReact","CollectionCurationReferrerTags","CollectionCurationReferrerParamNames"],(function a(b,c,d,e,f,g,h){'use strict';var i=c("reactRe").Component[10][7],j=i[1],k=i[2],l=i[3],m=i[5],n="MessengerSaveMessageDialogReact";function o(){return [0];}function p(y){var z=y[1];y[4][0]=[setTimeout(function(){return c("bs_curry")._1(z[0],false);},2000)];return 0;}function q(y){var z=y[4][0];if(z){clearTimeout(z[0]);return 0;}else return 0;}function r(y){var z={};z[c("CollectionCurationReferrerParamNames").CLICK_REF]=c("CollectionCurationReferrerTags").MESSENGER_DOT_COM_SAVE_CONFIRMATION;var aa=new (c("URI"))("https://www.facebook.com/saved/").setQueryData(z);return c("bs_curry").app(c("messengerDialogReact").Dialog[0],[0,y[1][0],0,0,0,0,0,[c("bs_curry")._6(c("messengerDialogReact").Header[0],0,0,[h._("\u6b63\u5728\u4fdd\u5b58\u5230 Facebook"),0],0,0,0),[c("bs_curry")._3(c("messengerDialogBodyReact").createElement(0,[c("linkReact").createElement(0,0,[aa],0,0,0,0,0,["_blank"],0)([h._("\u67e5\u770b\u6536\u85cf\u5939"),0],0,0,0),0]),0,0,0),0]],0,0,0]);}var s=[function(y){return [y.onToggle];}],t=[j,k,l,m,n,o,p,q,r,s],u=c("reactRe").CreateComponent([n,o,m,p,l,j,k,q,s,r]),v=u[1],w=c("bs_curry").__1(v),x=u[0];g.MessengerSaveMessageDialog=t;g.comp=x;g.wrapProps=v;g.createElement=w;}),null);
__d('MessengerMessageActions.react',['cx','fbt','FBClipboardLink.react','ImmutableObject','ReactLayeredComponentMixin_DEPRECATED','MercuryAttachmentType','MercuryConfig','MercuryMessageActions','MercuryMessageInfo','messengerDialogsRe','MessengerContextualActions.react','messengerDeleteMessageDialogReact','MessengerReactionsMenu.react','React','AsyncRequest','CollectionsDisplaySurface','CollectionCurationMechanisms','MessengerConfig','MessengerDotComSaveModule','messengerSaveMessageDialogReact','messengerURIUtilsRe','SaveMessageUtils','SavedStateActions','URLMatcher','XMessengerDotComSaveController'],(function a(b,c,d,e,f,g,h,i){'use strict';var j=c('messengerDeleteMessageDialogReact').comp,k=c('messengerSaveMessageDialogReact').comp,l=c('MessengerContextualActions.react').Item,m=c('React').PropTypes,n=c('React').createClass({displayName:'MessengerMessageActions',mixins:[c('ReactLayeredComponentMixin_DEPRECATED')],propTypes:{isActive:m.bool,isDeletable:m.bool,isFromViewer:m.bool,includeReactions:m.bool,message:m.instanceOf(c('ImmutableObject')).isRequired},getDefaultProps:function o(){return {isDeletable:true};},getInitialState:function o(){return {shown:false};},render:function o(){return this.props.isActive?c('React').createElement('span',{className:"_2rvp",ref:'actions',onClick:this._toggleShown},this.renderMenu()):null;},renderMenu:function o(){if(!this.state.open)return null;return c('React').createElement(c('MessengerReactionsMenu.react'),{entryPoint:'dot_menu',message:this.props.message,onHide:this._hidePopover,refProp:function(){return this.refs.actions;}.bind(this)});},_openPopover:function o(){this.setState({open:true,shown:false});},_hidePopover:function o(){this.setState({open:false});},renderLayers:function o(){if(!this.state.shown)return {};var p=[];p.push(c('React').createElement(l,{key:'delete',onClick:this._handleDelete},this._renderActionItemMsg()));if(this._canSaveMessage())p.push(c('React').createElement(l,{key:'save',onClick:this._handleSave},this._renderSaveItemMsg()));if(c('MessengerConfig').ShowMessageLinks)p.push(c('React').createElement(l,{key:'react',onClick:this._handleLink},this._renderLinkItemMsg()));if(this.props.includeReactions)p.push(c('React').createElement(l,{key:'perma-link',onClick:this._openPopover},this._renderReactionMsg()));return {contextualDialog:c('React').createElement(c('MessengerContextualActions.react'),{context:this.refs.actions,onToggle:this._handleActionsToggle,shown:this.props.isActive},this._renderRetry(),p)};},_toggleShown:function o(){this.setState({shown:!this.state.shown},function(){return this.props.onShowMenu(this.state.shown);}.bind(this));},_renderActionItemMsg:function o(){if(!this.props.isFromViewer&&c('MercuryConfig').MessengerGlobalDeleteGK)return i._("\u9690\u85cf");return i._("\u5220\u9664");},_renderSaveItemMsg:function o(){return i._("\u6536\u85cf\u5230 Facebook");},_renderReactionMsg:function o(){return i._("\u5fc3\u60c5");},_renderLinkItemMsg:function o(){var p=c('messengerURIUtilsRe').getURIForMessageID(this.props.message.message_id).getQualifiedURIBase().toString(),q=i._("\u590d\u5236\u94fe\u63a5");return c('React').createElement(c('FBClipboardLink.react'),{suppress:true,tooltip:'',value:p},q);},_renderRetry:function o(){if(!c('MercuryMessageInfo').hasError(this.props.message))return null;return c('React').createElement(l,{onClick:this._handleRetry},i._("\u91cd\u8bd5"));},_handleActionsToggle:function o(p){!p&&this._handleDeselect();},_handleDeselect:function o(){this.setState({shown:false},function(){return this.props.onShowMenu(this.state.shown);}.bind(this));},_handleDelete:function o(){var p=this.props.message;c('messengerDialogsRe').showDialog(function(){return c('React').createElement(j,{onDelete:function q(){c('MercuryMessageActions').get()['delete'](p.thread_id,[p.message_id]);},onToggle:this._handleDialogToggle,onCancel:this._handleDeselect,isFromViewer:this.props.isFromViewer});}.bind(this));},_canSaveMessage:function o(){var p=this.props.message;if(c('MessengerDotComSaveModule').enabled_messenger_save&&this._isSavableMessageAttachment())return true;if(c('MessengerDotComSaveModule').enabled_messenger_save&&p.attachments&&p.attachments.length>0&&p.attachments[0].share&&p.attachments[0].share.media&&p.attachments[0].share.media.playable)return true;if(c('MessengerDotComSaveModule').enabled_messenger_save){var q=c('URLMatcher').match(p.body);if(q!=null&&q.length>0)return true;}return false;},_handleLink:function o(){this._handleActionsToggle(false);},_isSavableMessageAttachment:function o(){var p=this.props.message.attachments;if(!p||p.length===0)return false;for(var q=0;q<p.length;q++){var r=p[q],s=false;if(r.share&&r.share.style_list){var t=r.share.style_list;s=c('SaveMessageUtils').isSavableMessageAttachment(t);}else s=r.attach_type===c('MercuryAttachmentType').VIDEO;if(!s)return false;}return true;},_handleSave:function o(){var p=this.props.message,q=c('CollectionsDisplaySurface').MESSENGER_DOT_COM_MESSAGE,r=c('CollectionCurationMechanisms').RIGHT_CLICK;new (c('AsyncRequest'))().setURI(c('XMessengerDotComSaveController').getURIBuilder().getURI()).setData({action:c('SavedStateActions').SAVE,surface:q,mechanism:r,message_id:p.message_id}).send();c('messengerDialogsRe').showDialog(function(){return c('React').createElement(k,{onToggle:this._handleDialogToggle});}.bind(this));this._handleActionsToggle(false);},_handleDialogToggle:function o(p){if(!p)c('messengerDialogsRe').removeDialog();},_handleRetry:function o(){this._handleDeselect();c('MercuryMessageActions').get().resend(this.props.message);}});f.exports=n;}),null);