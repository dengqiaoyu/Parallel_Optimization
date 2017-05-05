if (self.CavalryLogger) { CavalryLogger.start_js(["cOK1H"]); }

__d("MessengerPaymentProductType",[],(function a(b,c,d,e,f,g){f.exports={P2P:"p2p",PAGES_COMMERCE:"pages_commerce",NMOR_PAGES_COMMERCE:"nmor_pages_commerce",MESSENGER_COMMERCE:"messenger_commerce"};}),null);
__d("NFXStoryLocation",[],(function a(b,c,d,e,f,g){f.exports={FEED:"feed",VIDEO_LIST_CHANNEL:"video_list_channel",VIDEO_PLAYLIST_CHANNEL:"video_playlist_channel",WATCHED_VIDEOS_CHANNEL:"watched_videos_channel",EXPLORE_FEED:"explore_feed",PROFILE_SELF:"profile_self",PROFILE_SOMEONE_ELSE:"profile_someone_else",PAGE:"page",PERMALINK:"permalink",PHOTO_VIEWER:"photo_viewer",PAGE_REVIEW:"page_review",POPULAR_AT:"popular_at",CURATED_SECTION:"curated_section",GROUP:"group",EVENT:"event",SEARCH:"search",UNKNOWN:"unknown",PROFILE_REPORTING_FLOW:"profile_reporting_flow",HEAD_PUBLISHER_APP_MENTIONS_FEED:"head_publisher_app_mentions_feed",MESSENGER:"messenger",MESSENGER_MONTAGE_VIEWER:"messenger_montage_viewer",FUNDRAISER_PAGE_FEED:"fundraiser_page_feed",FUNDRAISER_PERSON_TO_CHARITY:"fundraiser_person_to_charity",FUNDRAISER_PERSON_FOR_PERSON:"fundraiser_person_for_person",SETTINGS:"settings",LIVE_MAP:"live_map",JOB_BROWSER:"job_browser",MESSENGER_THREAD_ACTION_PANEL:"messenger_thread_action_panel",MESSENGER_CONTACT_DETAILS:"messenger_contact_details",TICKER:"ticker",FULLSCREEN_VIDEO_PLAYER:"fullscreen_video_player",ACTIVITY_LOG_FACE_ALERTS:"face_alerts",HELP_COMMUNITY:"help_community",SUPPORT_INBOX:"support_inbox",APP_INVITES_FEED:"app_invites_feed",CHAINING_FEED:"chaining_feed",SOCIAL_REPORTING_REDIRECT:"social_reporting_redirect",WWW_CHAT_HEAD:"www_chat_head",THROWBACK:"throwback",POST_TO_PAGE:"post_to_page",ACTIVITY_LOG:"activity_log",VIDEO_CHANNEL:"video_channel",VIDEO_CHANNELS:"video_channels",VIDEO_HOME:"video_home",REVIEWS_FEED:"reviews_feed",BACKSTAGE:"backstage",MESSENGER_ENCRYPTED_THREAD:"messenger_encrypted_thread",LOCAL_NEWS:"local_news",MARKETPLACE_PDP:"marketplace_pdp",MARKETPLACE_PROFILE:"marketplace_profile",MARKETPLACE_RATING_CONFIRMATION:"marketplace_rating_confirmation",MARKETPLACE_RATING_FLOW:"marketplace_rating_flow",MARKETPLACE_THREAD:"marketplace_thread",MARKETPLACE_THREAD_BUYER:"marketplace_thread_buyer",POLITICAL_ISSUE_MODULE:"political_issue_module",ELECTION_HUB:"election_hub",DIRECT_MESSAGING:"direct_messaging",CAMERA_MEDIA_EFFECT:"camera_media_effect",SPATIAL_PRIVACY_PUBLIC:"spatial_privacy_public",SPATIAL_PRIVACY_FRIENDS:"spatial_privacy_friends",SPATIAL_PRIVACY_ALL:"spatial_privacy_all",SPATIAL_PRIVACY_SELF:"spatial_privacy_self",PROFILE_DEPRECATED:"profile",TIMELINE_DEPRECATED:"timeline",BREAKUP_FLOW:"breakup_flow",ASYNC_TASK_DO_NOT_USE:"async",INSTANT_ARTICLES:"instant_articles",IN_APP_BROWSER:"in_app_browser",INSTREAM_VIDEO:"instream_video",TOPIC_PAGE:"topic_page"};}),null);
__d('MessengerMenu.react',['cx','Keys','MenuSeparator.react','ReactXUIMenu','joinClasses'],(function a(b,c,d,e,f,g,h){'use strict';var i,j;function k(m){m.isReactLegacyFactory={};m.type=m;}i=babelHelpers.inherits(l,c('ReactXUIMenu'));j=i&&i.prototype;function l(m){var n=m.className,o=babelHelpers.objectWithoutProperties(m,['className']);j.constructor.call(this,babelHelpers['extends']({className:c('joinClasses')(n,"_2i-c _150g")},o));}l.prototype.handleKeydown=function(m,n){if(m===c('Keys').DOWN||m===c('Keys').UP||m===c('Keys').SPACE)return j.handleKeydown.call(this,m,n);return true;};k(l);l.Item=c('ReactXUIMenu').Item;l.Separator=c('MenuSeparator.react');f.exports=l;}),null);
__d('MessengerPopoverMenu.react',['cx','ContextualDialogArrow','ContextualLayerAutoFlip','ContextualLayerUpdateOnScroll','PopoverMenu.react','React','joinClasses'],(function a(b,c,d,e,f,g,h){'use strict';var i,j;i=babelHelpers.inherits(k,c('React').PureComponent);j=i&&i.prototype;k.prototype.render=function(){var l=this.props,m=l.children,n=l.className,o=l.isOpen,p=babelHelpers.objectWithoutProperties(l,['children','className','isOpen']),q=c('React').Children.only(m);return c('React').createElement(c('PopoverMenu.react'),babelHelpers['extends']({className:c('joinClasses')(n,!p.disableArrowKeyActivation||o?"_150g":''),enableActivationOnEnter:true,layerBehaviors:[c('ContextualLayerAutoFlip'),c('ContextualLayerUpdateOnScroll'),c('ContextualDialogArrow')]},p),q);};function k(){i.apply(this,arguments);}f.exports=k;}),null);
__d("XSettingsEmployeeBetaController",["XController"],(function a(b,c,d,e,f,g){f.exports=c("XController").create("\/settings\/employee\/beta_mode\/",{enabled:{type:"Bool",defaultValue:false}});}),null);
__d('MessengerBugNub.react',['cx','fbt','ix','AsyncRequest','Image.react','Link.react','MessengerEnvironment','MessengerMenu.react','MessengerPopoverMenu.react','React','Tooltip','URI','XSettingsEmployeeBetaController','goURI'],(function a(b,c,d,e,f,g,h,i,j){'use strict';var k,l,m=j("86836"),n=j("86835"),o=j("86837"),p=c('React').PropTypes,q=c('MessengerMenu.react').Item,r=c('MessengerMenu.react').Separator;k=babelHelpers.inherits(s,c('React').PureComponent);l=k&&k.prototype;s.prototype.render=function(){return c('React').createElement('div',{className:"_4_xe"},this.$MessengerBugNub1(),c('React').createElement(c('Link.react'),babelHelpers['extends']({href:'#',ajaxify:'/ajax/bugs/employee_report',className:"_1fr8",rel:'dialog'},c('Tooltip').propsFor(i._("\u62a5\u544a\u6f0f\u6d1e"))),c('React').createElement('div',{className:"_1gng"},c('React').createElement(c('Image.react'),{src:m}))));};s.prototype.$MessengerBugNub1=function(){var t=this.props.isBetaEnabled?n:o,u=i._("\u5185\u90e8\u9996\u9009\u9879");return c('React').createElement(c('MessengerPopoverMenu.react'),babelHelpers['extends']({className:"_1fr8",menu:this.$MessengerBugNub2()},c('Tooltip').propsFor(u)),c('React').createElement(c('Link.react'),{'aria-haspopup':'true','aria-label':u,className:"_1gng",href:'#',role:'button'},c('React').createElement(c('Image.react'),{src:t})));};s.prototype.$MessengerBugNub2=function(){var t,u=this,v=null;if(this.props.unixName)(function(){var w=u.props.unixName+'.sb';v=c('React').createElement(q,{label:'Sandbox',onclick:function(){return this.$MessengerBugNub3(w);}.bind(u)});})();return c('React').createElement(c('MessengerMenu.react'),null,c('React').createElement(q,{label:'Public',onclick:function(){return this.$MessengerBugNub4(false);}.bind(this)}),c('React').createElement(q,{label:'Beta',onclick:function(){return this.$MessengerBugNub4(true);}.bind(this)}),c('React').createElement(r,null),c('React').createElement(q,{label:'C1 (trunkstable)',onclick:function(){return this.$MessengerBugNub3('trunkstable');}.bind(this)}),c('React').createElement(q,{label:'Latest',onclick:function(){return this.$MessengerBugNub3('latest');}.bind(this)}),c('React').createElement(q,{label:'Intern',onclick:function(){return this.$MessengerBugNub3('intern');}.bind(this)}),c('React').createElement(q,{label:'Production',onclick:function(){return this.$MessengerBugNub3('prod');}.bind(this)}),v);};s.prototype.$MessengerBugNub4=function(t){var u=c('XSettingsEmployeeBetaController').getURIBuilder().getURI();new (c('AsyncRequest'))().setURI(u).setData({enabled:t}).setMethod('POST').setAllowCrossPageTransition(true).send();};s.prototype.$MessengerBugNub3=function(t){var u=c('URI').getRequestURI(),v=c('MessengerEnvironment').facebookdotcom?'.facebook.com':'.messenger.com',w='www'+(t?'.'+t:'')+v;c('goURI')(new (c('URI'))(window.location.href).setProtocol(u.getProtocol()).setDomain(w).setSubdomain(u.getSubdomain()).setPath(u.getPath()).setQueryData(u.getQueryData()).setFragment(u.getFragment()));};function s(){k.apply(this,arguments);}s.propTypes={isBetaEnabled:p.bool,unixName:p.string};f.exports=s;}),null);
__d('MessengerRTCGroupCallContactListItem.react',['cx','fbt','ix','Image.react','Link.react','messengerContactImageReact','MultiwayTypes','React','emptyFunction'],(function a(b,c,d,e,f,g,h,i,j){'use strict';var k,l=c('messengerContactImageReact').comp,m=c('MultiwayTypes').ParticipantCallState,n=(k={},k[m.DISCONNECTED]=i._("\u5df2\u65ad\u5f00\u8fde\u63a5"),k[m.NO_ANSWER]=i._("\u672a\u63a5\u542c"),k[m.REJECTED]=i._("\u88ab\u62d2\u7edd"),k[m.UNREACHABLE]=i._("\u65e0\u6cd5\u63a5\u901a"),k[m.CONNECTION_DROPPED]=i._("\u5df2\u6389\u7ebf"),k[m.CONTACTING]=i._("\u6b63\u5728\u63a5\u901a..."),k[m.RINGING]=i._("\u6b63\u5728\u547c\u53eb\u2026"),k[m.CONNECTING]=i._("\u6b63\u5728\u8fde\u63a5..."),k[m.CONNECTED]=i._("\u5df2\u8fde\u63a5"),k[m.PARTICIPANT_LIMIT_REACHED]=i._("\u53c2\u4e0e\u4eba\u6570\u8fbe\u5230\u663e\u793a"),k[m.IN_ANOTHER_CALL]=i._("\u901a\u8bdd\u4e2d"),k),o=32,p=function t(u){var v=u.isSelected,w=u.onClick,x=u.participant,y=u.participantState;return c('React').createElement(c('Link.react'),{'aria-checked':v,className:"_4nvn",onClick:s(y)?c('emptyFunction'):w,role:'checkbox'},c('React').createElement(l,{className:"_4nvr",isMessengerUser:x.is_messenger_user,size:o,src:x.image_src}),c('React').createElement('div',{className:"_4nvs"},c('React').createElement('div',{className:"_4nvt"},c('React').createElement('div',{className:"_4nv_"},x.name),c('React').createElement(q,{participant:x,participantState:y})),c('React').createElement(r,{isSelected:v,participantState:y})));},q=function t(u){var v=u.participant.vanity,w=u.participantState;if(w){var x=n[w.state]||i._("\u672a\u77e5");return c('React').createElement('div',{className:"_4nw0"},x);}if(v)return c('React').createElement('div',{className:"_4nw0"},'@',v);return null;},r=function t(u){var v=u.isSelected,w=u.participantState;if(s(w)){return c('React').createElement('div',{className:"_1j79"},i._("\u901a\u8bdd\u4e2d"));}else if(v)return c('React').createElement('div',{className:"_1j79"},c('React').createElement(c('Image.react'),{src:j("86852")}));return null;};function s(t){if(!t)return false;switch(t.state){case m.CONTACTING:case m.RINGING:case m.CONNECTING:case m.CONNECTED:return true;default:return false;}}f.exports=p;}),null);
__d('MessengerRTCGroupCallContactList.react',['cx','immutable','MessengerRTCGroupCallContactListItem.react','MessengerScrollableArea.react','React'],(function a(b,c,d,e,f,g,h){'use strict';var i=function j(k){var l=k.isParticipantSelected,m=k.onClick,n=k.participants,o=k.participantStates,p=o===undefined?c('immutable').Map():o,q=n.map(function(r){return c('React').createElement(c('MessengerRTCGroupCallContactListItem.react'),{isSelected:l.get(r.id),key:r.id,onClick:function(s){function t(){return s.apply(this,arguments);}t.toString=function(){return s.toString();};return t;}(function(){return m(r.id);}),participant:r,participantState:p.get(r.fbid||'')});});return c('React').createElement(c('MessengerScrollableArea.react'),{className:"_12zw"},q);};f.exports=i;}),null);
__d('MessengerRTCGroupCallRingParticipantsRow.react',['cx','fbt','React','intlList'],(function a(b,c,d,e,f,g,h,i){'use strict';var j=function k(l){var m=l.names;return c('React').createElement('div',{className:"_1wsd"},c('React').createElement('div',{className:"_1wse"},i._("\u547c\u53eb\uff1a")),c('React').createElement('div',{className:"_1wsg"},c('intlList')(m,c('intlList').CONJUNCTIONS.NONE)));};f.exports=j;}),null);
__d('MessengerRTCGroupCallThreadRow.react',['cx','fbt','MercuryThreadTitle.react','messengerThreadImageReact','React'],(function a(b,c,d,e,f,g,h,i){'use strict';var j=c('messengerThreadImageReact').comp,k=function l(m){var n=m.displayAddMemberWarning,o=m.participants,p=m.thread,q=m.viewerID;return c('React').createElement('div',{className:"_5y31"},c('React').createElement(j,{className:"_5y32",participants:o,thread:p,viewer:q}),c('React').createElement('div',{className:"_5y34"},c('React').createElement(c('MercuryThreadTitle.react'),{className:"_5y37",thread:p,viewer:q}),n?c('React').createElement('div',{className:"_1apf"},i._("\u4f60\u6dfb\u52a0\u5230\u89c6\u9891\u7fa4\u804a\u7684\u7528\u6237\u5c06\u770b\u5230\u5bf9\u8bdd\u8bb0\u5f55\u3002")):null));};f.exports=k;}),null);
__d('MessengerRTCGroupCallParticipantsPickerDialog.react',['cx','fbt','FBRTCCore','immutable','ImmutableObject','MessengerDialog.react','MessengerDialogButton.react','MessengerDialogCancelButton.react','MessengerDialogHeader.react','MessengerDialogTitle.react','MessengerRTCGroupCallContactList.react','MessengerRTCGroupCallRingParticipantsRow.react','MessengerRTCGroupCallThreadRow.react','React'],(function a(b,c,d,e,f,g,h,i){'use strict';var j,k,l=c('React').PropTypes,m=460,n=5;j=babelHelpers.inherits(o,c('React').Component);k=j&&j.prototype;function o(){var p,q;for(var r=arguments.length,s=Array(r),t=0;t<r;t++)s[t]=arguments[t];return q=(p=k.constructor).call.apply(p,[this].concat(s)),this.state={isParticipantSelected:this.props.participants.mapEntries(function(u){var v=u[0],w=u[1];return [v,this.props.participants.size<=n];}.bind(this))},this.$MessengerRTCGroupCallParticipantsPickerDialog1=function(){var u=this.$MessengerRTCGroupCallParticipantsPickerDialog2().map(function(v){return v.fbid;});if(this.props.handleCall){this.props.handleCall(u);}else c('FBRTCCore').startGroupCall({conferenceName:this.props.conferenceName,hasVideo:this.props.hasVideo,trigger:this.props.trigger,usersToRing:u});this.props.onUnmount();}.bind(this),q;}o.prototype.render=function(){var p=this.props,q=p.participants,r=p.thread,s=p.viewerID;return c('React').createElement(c('MessengerDialog.react'),{onToggle:this.props.onUnmount,type:'default',width:m},c('React').createElement('div',{className:"_2m1r"},c('React').createElement(c('MessengerDialogHeader.react'),null,c('React').createElement(c('MessengerDialogCancelButton.react'),null),c('React').createElement(c('MessengerDialogTitle.react'),null,i._("\u547c\u53eb\u7fa4\u804a\u6210\u5458")),c('React').createElement(c('MessengerDialogButton.react'),{label:i._("\u547c\u53eb"),onClick:this.$MessengerRTCGroupCallParticipantsPickerDialog1,disabled:this.$MessengerRTCGroupCallParticipantsPickerDialog2().length===0,type:'primary'})),c('React').createElement(c('MessengerRTCGroupCallThreadRow.react'),{participants:q,thread:r,viewerID:s}),c('React').createElement(c('MessengerRTCGroupCallRingParticipantsRow.react'),{names:this.$MessengerRTCGroupCallParticipantsPickerDialog2().map(function(t){var u=t.name;return u;}).sort(function(t,u){return t.localeCompare(u);})}),c('React').createElement(c('MessengerRTCGroupCallContactList.react'),{participants:q.toArray().sort(function(t,u){return t.name.localeCompare(u.name);}),isParticipantSelected:this.state.isParticipantSelected,onClick:function(t){return this.setState({isParticipantSelected:this.state.isParticipantSelected.set(t,!this.state.isParticipantSelected.get(t))});}.bind(this)})));};o.prototype.$MessengerRTCGroupCallParticipantsPickerDialog2=function(){return this.props.participants.filter(function(p,q){return this.state.isParticipantSelected.get(q);}.bind(this)).toArray();};o.propTypes={participants:l.instanceOf(c('immutable').Map).isRequired,thread:l.instanceOf(c('ImmutableObject')).isRequired,viewerID:l.string.isRequired,trigger:l.string.isRequired,conferenceName:l.string.isRequired,hasVideo:l.bool.isRequired,handleCall:l.func,onUnmount:l.func.isRequired};f.exports=o;}),null);
__d('MessengerRTCGroupCallParticipantsPickerDialogController',['DOM','immutable','MessengerRTCGroupCallParticipantsPickerDialog.react','React','ReactDOM'],(function a(b,c,d,e,f,g){'use strict';var h={_container:c('DOM').create('div'),render:function i(j){var k=j.participants,l=j.thread,m=j.viewerID,n=j.trigger,o=j.conferenceName,p=j.hasVideo,q=j.handleCall;c('ReactDOM').render(c('React').createElement(c('MessengerRTCGroupCallParticipantsPickerDialog.react'),{participants:k,thread:l,viewerID:m,trigger:n,conferenceName:o,hasVideo:p,handleCall:q,onUnmount:function(){return this._onUnmount();}.bind(this)}),this._container);},_onUnmount:function i(){c('ReactDOM').unmountComponentAtNode(this._container);}};f.exports=h;}),null);
__d('P2PPaymentRequestsStore',['P2PActionConstants','Arbiter','ChannelConstants','CurrentUser','EventEmitter','immutable','ImmutableObject','P2PActions','P2PAPI','P2PChannelType','P2PDispatcher','P2PPaymentRequest','P2PPaymentRequestStatus'],(function a(b,c,d,e,f,g){'use strict';var h,i,j=c('immutable').List,k=null,l=false,m=null,n=false,o=null,p=new j(),q=false,r=false;h=babelHelpers.inherits(s,c('EventEmitter'));i=h&&h.prototype;function s(){i.constructor.call(this);c('Arbiter').subscribe(c('ChannelConstants').getArbiterType(c('P2PChannelType').PAYMENT_REQUEST_STATUS_CHANGED),this.handleChannelPaymentRequestStatusChanged.bind(this));c('P2PDispatcher').register(this.onEventDispatched.bind(this));}s.prototype.onEventDispatched=function(t){var u=t.data,v=t.type;switch(v){case c('P2PActionConstants').PAYMENT_REQUEST_INITIATED:q=true;o=null;this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_INITIATED_COMPLETE:q=false;this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_CREATED:this.handlePaymentRequestCreated(u);this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_CREATED_ERROR:o=new (c('ImmutableObject'))(u.error);this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_UPDATED:this.handlePaymentRequestUpdated(u);this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_DECLINE_INITIATED:n=true;m=null;this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_DECLINED:n=false;m=null;this.handlePaymentRequestDeclined(u);this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_DECLINE_ERROR:n=false;m=new (c('ImmutableObject'))(u.error);this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_CANCEL_INITIATED:l=true;k=null;this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_CANCELED:l=false;k=null;this.handlePaymentRequestCanceled(u);this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUEST_CANCEL_ERROR:l=false;k=new (c('ImmutableObject'))(u.error);this.emit('change');break;case c('P2PActionConstants').DIALOG_CLOSED:m=null;this.emit('change');break;case c('P2PActionConstants').PAYMENT_REQUESTS_FETCHED:this.handlePaymentRequestsFetched(u);this.emit('change');break;}};s.prototype.handlePaymentRequestCreated=function(t){var u=p.toArray();u.push(new (c('P2PPaymentRequest'))(t));u.sort(function(v,w){return v.creationTime-w.creationTime;});p=new j(u);};s.prototype.handlePaymentRequestUpdated=function(t){if(t.groupRequestID)this.updateIndividualRequest(t);delete t.groupRequestID;this.updateRequest(t.id,t);};s.prototype.handlePaymentRequestDeclined=function(t){this.updateRequest(t,{currentStatus:c('P2PPaymentRequestStatus').DECLINED});};s.prototype.handlePaymentRequestCanceled=function(t){this.updateRequest(t,{currentStatus:c('P2PPaymentRequestStatus').CANCELED});};s.prototype.handleChannelPaymentRequestStatusChanged=function(t,u){var v=u.obj;c('P2PActions').paymentRequestUpdated({id:v.id,currentStatus:v.current_status,groupRequestID:v.group_request_id,transfer:v.transfer});};s.prototype.handlePaymentRequestsFetched=function(t){this.$P2PPaymentRequestsStore1(t);};s.prototype.$P2PPaymentRequestsStore1=function(t){t=t.sort(function(v,w){return v.creationTime-w.creationTime;});t.forEach(function(v,w){if(!this.getPaymentRequestByID(v.id))p=p.push(new (c('P2PPaymentRequest'))(v));}.bind(this));var u=p.toArray();u.sort(function(v,w){return v.creationTime-w.creationTime;});p=new j(u);};s.prototype.isPaymentRequestInitiated=function(){return q;};s.prototype.isPaymentRequestDeclining=function(){return n;};s.prototype.isPaymentRequestCanceling=function(){return l;};s.prototype.getPaymentRequestByID=function(t){return p.find(function(u){return t===u.id;});};s.prototype.getPaymentRequestError=function(){return o;};s.prototype.getPaymentRequestDeclineError=function(){return m;};s.prototype.getPaymentRequestCancelError=function(){return k;};s.prototype.getPendingPaymentRequest=function(){if(!r){r=true;c('P2PAPI').getPendingPaymentRequests();}return p.filter(function(t){return t.requestee.id===c('CurrentUser').getID()&&t.currentStatus===c('P2PPaymentRequestStatus').INITED;}).last();};s.prototype.updateRequest=function(t,u){p.forEach(function(v,w){if(t===v.id)p=p.set(w,v.merge(v,u));});};s.prototype.updateIndividualRequest=function(t){var u=t.id,v=t.groupRequestID,w=p.findIndex(function(ca){return v===ca.id;}),x=p.get(w),y=x.individualRequests.findIndex(function(ca){return u===ca.id;}),z=x.individualRequests.slice(0),aa=z[y],ba=babelHelpers['extends']({},aa,t);z[y]=ba;x=x.set('individualRequests',z);p=p.set(w,x);};s.prototype.getPaymentRequests=function(){return p;};f.exports=new s();}),null);
__d('P2PSimpleDialog.react',['fbt','FBPaymentsErrorNotice_DEPRECATED.react','P2PButton.react','P2PDialog.react','P2PDialogBody.react','P2PDialogFooter.react','P2PDialogTitle.react','P2PLoadingMask.react','React'],(function a(b,c,d,e,f,g,h){'use strict';var i,j,k=c('React').PropTypes;i=babelHelpers.inherits(l,c('React').Component);j=i&&i.prototype;function l(){var m,n;for(var o=arguments.length,p=Array(o),q=0;q<o;q++)p[q]=arguments[q];return n=(m=j.constructor).call.apply(m,[this].concat(p)),this.handleToggle=function(r){if(!r)this.props.onCancel();}.bind(this),n;}l.prototype.render=function(){var m=void 0;if(this.props.error)m=c('React').createElement(c('FBPaymentsErrorNotice_DEPRECATED.react'),{error:this.props.error});return c('React').createElement(c('P2PDialog.react'),{behaviors:this.props.behaviors,layerHideOnBlur:false,onToggle:this.handleToggle,repositionOnUpdate:true,shown:true,width:this.props.width},c('React').createElement(c('P2PDialogTitle.react'),null,this.props.title),c('React').createElement(c('P2PDialogBody.react'),null,m,this.props.bodyText,c('React').createElement(c('P2PLoadingMask.react'),{visible:this.props.loading})),c('React').createElement(c('P2PDialogFooter.react'),null,c('React').createElement(c('P2PButton.react'),{disabled:this.props.loading,label:this.props.cancelButtonText,onClick:this.props.onCancel,size:'medium'}),c('React').createElement(c('P2PButton.react'),{disabled:this.props.loading,label:this.props.confirmButtonText,onClick:this.props.onConfirm,size:'medium',use:'confirm'})));};l.propTypes={bodyText:k.string.isRequired,cancelButtonText:k.string,confirmButtonText:k.string,error:k.object,loading:k.bool,onCancel:k.func.isRequired,onConfirm:k.func.isRequired,title:k.string.isRequired,width:k.number};l.defaultProps={cancelButtonText:h._("\u53d6\u6d88"),confirmButtonText:h._("\u786e\u8ba4"),width:300};f.exports=l;}),null);
__d('P2PPaymentRequestCancelDialogContainer.react',['fbt','P2PActions','P2PAPI','P2PLogger','P2PPaymentLoggerEvent','P2PPaymentLoggerEventFlow','P2PPaymentRequest','P2PPaymentRequestsStore','P2PSimpleDialog.react','React','StoreAndPropBasedStateMixin'],(function a(b,c,d,e,f,g,h){'use strict';var i=c('React').PropTypes,j=c('React').createClass({displayName:'P2PPaymentRequestCancelDialogContainer',mixins:[c('StoreAndPropBasedStateMixin')(c('P2PPaymentRequestsStore'))],propTypes:{paymentRequest:i.instanceOf(c('P2PPaymentRequest')).isRequired},statics:{calculateState:function k(l){return {error:c('P2PPaymentRequestsStore').getPaymentRequestCancelError(),loading:c('P2PPaymentRequestsStore').isPaymentRequestCanceling()};}},log:function k(l,m){c('P2PLogger').log(l,{www_event_flow:c('P2PPaymentLoggerEventFlow').UI_FLOW_P2P_REQUEST,request_id:this.props.paymentRequest.id});},componentDidMount:function k(){this.log(c('P2PPaymentLoggerEvent').UI_ACTN_INITIATE_CANCEL_REQUEST);},handleCancel:function k(){c('P2PActions').hideDialog();},handleConfirm:function k(){c('P2PAPI').cancelPaymentRequest({error:function(){this.log(c('P2PPaymentLoggerEvent').UI_ACTN_CANCEL_REQUEST_FAIL);}.bind(this),paymentRequestID:this.props.paymentRequest.id,success:function(){this.log(c('P2PPaymentLoggerEvent').UI_ACTN_CANCEL_REQUEST_SUCCESS);}.bind(this)});this.log(c('P2PPaymentLoggerEvent').UI_ACTN_CONFIRM_CANCEL_REQUEST);},renderBodyText:function k(){var l=this.props.paymentRequest.requestee.name;return h._("{Name of requestee}\u5c06\u6536\u5230\u8bf7\u6c42\u88ab\u62d2\u7684\u901a\u77e5\u3002",[h.param('Name of requestee',l)]);},render:function k(){return c('React').createElement(c('P2PSimpleDialog.react'),{bodyText:this.renderBodyText(),error:this.state.error,loading:this.state.loading,cancelButtonText:h._("\u8fd4\u56de"),confirmButtonText:h._("\u53d6\u6d88"),onCancel:this.handleCancel,onConfirm:this.handleConfirm,title:h._("\u53d6\u6d88\u8bf7\u6c42\uff1f")});}});f.exports=j;}),null);
__d('P2PPaymentRequestDeclineDialogContainer.react',['fbt','P2PActions','P2PAPI','P2PLogger','P2PPaymentLoggerEvent','P2PPaymentLoggerEventFlow','P2PPaymentRequest','P2PPaymentRequestsStore','P2PSimpleDialog.react','React','StoreAndPropBasedStateMixin'],(function a(b,c,d,e,f,g,h){'use strict';var i=c('React').PropTypes,j=c('React').createClass({displayName:'P2PPaymentRequestDeclineDialogContainer',mixins:[c('StoreAndPropBasedStateMixin')(c('P2PPaymentRequestsStore'))],propTypes:{paymentRequest:i.instanceOf(c('P2PPaymentRequest')).isRequired},statics:{calculateState:function k(l){return {error:c('P2PPaymentRequestsStore').getPaymentRequestDeclineError(),loading:c('P2PPaymentRequestsStore').isPaymentRequestDeclining()};}},log:function k(l,m){c('P2PLogger').log(l,{www_event_flow:c('P2PPaymentLoggerEventFlow').UI_FLOW_P2P_REQUEST,request_id:this.props.paymentRequest.id});},componentDidMount:function k(){this.log(c('P2PPaymentLoggerEvent').UI_ACTN_INITIATE_DECLINE_REQUEST);},handleCancel:function k(){c('P2PActions').hideDialog();},handleConfirm:function k(){c('P2PAPI').declinePaymentRequest({error:function(){this.log(c('P2PPaymentLoggerEvent').UI_ACTN_DECLINE_REQUEST_FAIL);}.bind(this),paymentRequestID:this.props.paymentRequest.id,success:function(){this.log(c('P2PPaymentLoggerEvent').UI_ACTN_DECLINE_REQUEST_SUCCESS);}.bind(this)});this.log(c('P2PPaymentLoggerEvent').UI_ACTN_CONFIRM_DECLINE_REQUEST);},renderBodyText:function k(){var l=this.props.paymentRequest.requester.name;return h._("{Name of requester}\u5c06\u6536\u5230\u6536\u6b3e\u8bf7\u6c42\u88ab\u62d2\u7684\u901a\u77e5\u3002",[h.param('Name of requester',l)]);},render:function k(){return c('React').createElement(c('P2PSimpleDialog.react'),{bodyText:this.renderBodyText(),error:this.state.error,loading:this.state.loading,cancelButtonText:h._("\u8fd4\u56de"),confirmButtonText:h._("\u62d2\u7edd"),onCancel:this.handleCancel,onConfirm:this.handleConfirm,title:h._("\u62d2\u7edd\u8bf7\u6c42\uff1f")});}});f.exports=j;}),null);
__d('P2PPaymentRequestActionHelper',['fbt','MercuryIDs','P2PActions','P2PPaymentRequest','P2PPaymentRequestCancelDialogContainer.react','P2PPaymentRequestStatus','P2PPaymentRequestDeclineDialogContainer.react'],(function a(b,c,d,e,f,g,h){'use strict';var i={initDeclinePaymentRequestFlow:function j(k){c('P2PActions').showDialog(c('P2PPaymentRequestDeclineDialogContainer.react'),{paymentRequest:k});},initPayForPaymentRequestFlow:function j(k,l){var m=[],n=k.groupThreadFBID,o=void 0;if(n){m.push(k.requester.id);o=c('MercuryIDs').getThreadIDFromThreadFBID(n);}else o=c('MercuryIDs').getThreadIDFromUserID(k.requester.id);c('P2PActions').chatSendViewOpened({amount:k.amount,memoText:k.memoText,paymentRequestID:k.id,threadID:o,useModal:true,referrer:l,groupSendRecipientUserIDs:m,groupThreadFBID:n});},initCancelPaymentRequestFlow:function j(k){c('P2PActions').showDialog(c('P2PPaymentRequestCancelDialogContainer.react'),{paymentRequest:k});},getStatusText:function j(k){var l=k.currentStatus,m=h._("\u5df2\u53d6\u6d88"),n=h._("\u5df2\u4ed8\u6b3e"),o=h._("\u5df2\u62d2\u7edd"),p=h._("\u672a\u4ed8\u6b3e"),q=void 0;switch(l){case c('P2PPaymentRequestStatus').DECLINED:q=o;break;case c('P2PPaymentRequestStatus').CANCELED:case c('P2PPaymentRequestStatus').TRANSFER_FAILED:q=m;break;case c('P2PPaymentRequestStatus').TRANSFER_COMPLETED:q=n;break;default:q=p;break;}return q;}};f.exports=i;}),null);
__d('P2PAcceptMoneyDialog.react',['cx','BootloadedComponent.react','Bootloader','MessengerEnvironment','P2PDialog.react','P2PLoadingMask.react','JSResource','React','Run'],(function a(b,c,d,e,f,g,h){'use strict';var i,j,k=c('React').PropTypes,l=c('MessengerEnvironment').messengerui;i=babelHelpers.inherits(m,c('React').Component);j=i&&i.prototype;function m(){var n,o;for(var p=arguments.length,q=Array(p),r=0;r<p;r++)q[r]=arguments[r];return o=(n=j.constructor).call.apply(n,[this].concat(q)),this.$P2PAcceptMoneyDialog1=function(){return c('React').createElement(c('P2PDialog.react'),{className:"_309"+(l?' '+"_5ktw":''),layerHideOnBlur:false,repositionOnUpdate:true,shown:true,width:l?400:350},c('React').createElement('div',{className:"_3w2b"},c('React').createElement(c('P2PLoadingMask.react'),{visible:true})));},o;}m.preload=function(){c('Run').onAfterLoad(function(){c('Bootloader').loadModules(["P2PAcceptMoneyDialogImpl.react"],function(){},'P2PAcceptMoneyDialog.react');});};m.prototype.render=function(){return c('React').createElement(c('BootloadedComponent.react'),babelHelpers['extends']({bootloadLoader:c('JSResource')('P2PAcceptMoneyDialogImpl.react').__setRef('P2PAcceptMoneyDialog.react'),bootloadPlaceholder:this.$P2PAcceptMoneyDialog1()},this.props));};m.propTypes={creditCards:k.array.isRequired,useRedesignForm:k.bool,onClose:k.func,transfer:k.object};f.exports=m;}),null);
__d('P2PVerificationFlowHelper',['AsyncDialog','AsyncRequest','P2PAPI','P2PTransferParam','emptyFunction'],(function a(b,c,d,e,f,g){var h=null,i=null,j=null,k={startVerificationFlow:function l(m,n,o){h=m;i=n;j=o||c('emptyFunction');c('AsyncDialog').send(new (c('AsyncRequest'))('/p2p/verify/dialog/?id='+m));},setupFlowExitHandler:function l(m,n,o,p){var q=function t(u){j(u);h=null;i=null;j=null;m.destroy();n.destroy();};if(o){m.subscribe('hide',function(){q(p);});}else{var r=false,s=false;m.subscribe('confirm',function(){r=true;});m.subscribe('hide',function(){if(r){m.destroy();n.destroy();}else if(i){n.show();}else q(false);});n.subscribe('confirm',function(){s=true;var t={};t[c('P2PTransferParam').TRANSFER_ID]=h;c('P2PAPI').cancelTransaction({formData:t});q(false);});n.subscribe('hide',function(){if(!s)m.show();});}}};f.exports=k;}),null);
__d('P2PCreditCardStore',['P2PActionConstants','Arbiter','CreditCardFormParam','ChannelConstants','EventEmitter','ImmutableObject','MessengerPaymentProductType','P2PActions','P2PAPI','P2PChannelType','P2PCreditCard','P2PDispatcher'],(function a(b,c,d,e,f,g){'use strict';var h,i,j=void 0,k=void 0,l=void 0,m=void 0,n=void 0,o=void 0,p=null,q=false,r=false;function s(v,w){if(v)o[v]=new (c('ImmutableObject'))(w);}function t(v,w){p=w;if(v)o[v]={error:w};}h=babelHelpers.inherits(u,c('EventEmitter'));i=h&&h.prototype;function u(){i.constructor.call(this);l=false;m=false;j=[];n=true;o={};k=c('P2PDispatcher').register(this.onEventDispatched.bind(this));c('Arbiter').subscribe(c('ChannelConstants').getArbiterType(c('P2PChannelType').CREDIT_CARD_CHANGED),this.handleChannelCreditCardChanged.bind(this));c('Arbiter').subscribe(c('ChannelConstants').getArbiterType(c('P2PChannelType').CREDIT_CARD_DELETED),this.handleChannelCreditCardDeleted.bind(this));}u.prototype.onEventDispatched=function(v){var w=v.data,x=v.type;switch(x){case c('P2PActionConstants').CREDIT_CARD_SAVING:this.handleCreditCardSaving();this.emit('change');break;case c('P2PActionConstants').CREDIT_CARDS_UPDATED:this.handleCreditCardsUpdated(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARDS_UPDATED_ERROR:this.handleCreditCardsUpdatedError();this.emit('change');break;case c('P2PActionConstants').CHANNEL_EVENTS_ALLOWED:this.handleChannelEventsAllowed();break;case c('P2PActionConstants').CHANNEL_EVENTS_IGNORED:this.handleChannelEventsIgnored();break;case c('P2PActionConstants').CREDIT_CARD_ADDED:this.handleCreditCardAdded(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_ADDED_ERROR:this.handleCreditCardAddedError(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_ADDED_ERROR_CLEARED:this.handleCreditCardAddedErrorCleared();this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_DELETED:this.handleCreditCardDeleted(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_DELETED_ERROR:this.handleCreditCardDeletedError(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_UPDATED:this.handleCreditCardUpdated(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_UPDATED_ERROR:this.handleCreditCardUpdatedError(w);this.emit('change');break;case c('P2PActionConstants').PRESET_CREDIT_CARD_UPDATED:this.handlePresetCreditCardUpdated(w);this.emit('change');break;case c('P2PActionConstants').PRESET_CREDIT_CARD_UPDATED_ERROR:this.handlePresetCreditCardUpdatedError(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_VERIFIED:this.handleCreditCardVerified(w);this.emit('change');break;case c('P2PActionConstants').CREDIT_CARD_VERIFIED_ERROR:this.handleCreditCardVerifiedError(w);this.emit('change');break;case c('P2PActionConstants').BIN_NUMBER_VALIDATED:this.handleBINNumberValidated(w);this.emit('change');break;case c('P2PActionConstants').BIN_NUMBER_VALIDATED_ERROR:this.handleBINNumberValidatedError(w);this.emit('change');break;}};u.prototype.handleCreditCardSaving=function(){q=true;p=null;};u.prototype.getAsyncRequestState=function(){return o;};u.prototype.isCreditCardsFetchComplete=function(){return l;};u.prototype.handleCreditCardsUpdated=function(v){l=true;r=false;j=v.map(function(w){return new (c('P2PCreditCard'))(w);});};u.prototype.handleCreditCardsUpdatedError=function(){l=true;r=true;};u.prototype.handleCreditCardAdded=function(v){q=false;s(v.requestID,v);if(!this.getCreditCardByCredentialID(v[c('CreditCardFormParam').CREDENTIAL_ID]))j.push(new (c('P2PCreditCard'))(v));};u.prototype.handleChannelEventsIgnored=function(){n=false;};u.prototype.handleChannelEventsAllowed=function(){n=true;};u.prototype.handleChannelCreditCardChanged=function(v,w){if(n&&!this.getCreditCardByCredentialID(w[c('CreditCardFormParam').CREDENTIAL_ID]))c('P2PAPI').getAllCreditCards();};u.prototype.handleCreditCardAddedError=function(v){q=false;t(v.requestID,v.errors);};u.prototype.handleCreditCardAddedErrorCleared=function(){p=null;};u.prototype.handleCreditCardDeleted=function(v){var w=v[c('CreditCardFormParam').CREDENTIAL_ID];s(v.requestID,v);j=j.filter(function(x){return x.getCredentialId()!==w;});};u.prototype.handleChannelCreditCardDeleted=function(v,w){w=w.obj;c('P2PActions').deleteCreditCard(w);};u.prototype.handleCreditCardDeletedError=function(v){t(v.requestID,v.error);};u.prototype.handleCreditCardUpdated=function(v){s(v.requestID,v);var w=this.getCreditCardByCredentialID(v[c('CreditCardFormParam').CREDENTIAL_ID]);if(w){w.setExp(v[c('CreditCardFormParam').CARD_EXPIRATION]);w.setZipCode(v[c('CreditCardFormParam').ZIP]);}};u.prototype.handleCreditCardUpdatedError=function(v){t(v.requestID,v.errors);};u.prototype.handlePresetCreditCardUpdated=function(v){var w=v[c('CreditCardFormParam').CREDENTIAL_ID];s(v.requestID,v);var x=j.filter(function(z){return z.getIsPreset();})[0];if(x)x.setIsPreset(false);var y=this.getCreditCardByCredentialID(w);if(y)y.setIsPreset(true);};u.prototype.handlePresetCreditCardUpdatedError=function(v){t(v.requestID,v.error);};u.prototype.handleCreditCardVerified=function(v){s(v.requestID,v);var w=this.getCreditCardByCredentialID(v[c('CreditCardFormParam').CREDENTIAL_ID]);if(w)w.setIsVerified(true);};u.prototype.handleCreditCardVerifiedError=function(v){t(v.requestID,v.error);};u.prototype.handleBINNumberValidated=function(v){s(v.requestID,v);};u.prototype.handleBINNumberValidatedError=function(v){t(v.requestID,v.error);};u.prototype.getAll=function(v){v=v===undefined?c('MessengerPaymentProductType').P2P:v;if(!m){m=true;c('P2PAPI').getAllCreditCards();}return j.filter(function(w){if(v===c('MessengerPaymentProductType').P2P)return w.getIsPersonalTransferEligible();return w;});};u.prototype.getCreditCardByCredentialID=function(v){return j.filter(function(w){return w.getCredentialId()===v;})[0];};u.prototype.getDispatchToken=function(){return k;};u.prototype.getSaveErrors=function(){return p;};u.prototype.isSaving=function(){return q;};u.prototype.failedCreditCardFetch=function(){return r;};f.exports=new u();}),null);
__d('CssBackgroundImage.react',['EncryptedImg','React','createCancelableFunction','cssURL'],(function a(b,c,d,e,f,g){var h=c('React').PropTypes,i=c('React').createClass({displayName:'CssBackgroundImage',getProps:{imageURI:h.string.isRequired,className:h.string,backgroundPosition:h.string,height:h.string,width:h.string,style:h.object,onMouseDown:h.func,onMouseMove:h.func,onMouseUp:h.func,onMouseOut:h.func},getInitialState:function j(){return {image:'data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACwAAAAAAQABAAACAkQBADs='};},getDefaultProps:function j(){return {backgroundPosition:'0% 0%',style:{}};},_process:function j(k){if(!c('EncryptedImg').isEncrypted(k)){this.setState({image:k});return;}if(this.encryptedImgCallback)this.encryptedImgCallback.cancel();this.encryptedImgCallback=c('createCancelableFunction')(function(l){if(k===this.props.imageURI)this.setState({image:l});}.bind(this));c('EncryptedImg').load(k,this.encryptedImgCallback);},componentWillMount:function j(){if(this.props.imageURI!=null)this._process(this.props.imageURI);},componentWillReceiveProps:function j(k){if(k.imageURI!=null)this._process(k.imageURI);},componentWillUnmount:function j(){if(this.encryptedImgCallback)this.encryptedImgCallback.cancel();},render:function j(){var k=this.props,l=k.imageURI,m=k.backgroundPosition,n=k.height,o=k.width,p=k.style,q=babelHelpers.objectWithoutProperties(k,['imageURI','backgroundPosition','height','width','style']);return c('React').createElement('div',babelHelpers['extends']({style:Object.assign({},p,{backgroundImage:c('cssURL')(this.state.image),backgroundPosition:m||p.backgroundPosition,height:n||p.height,width:o||p.width})},q));}});f.exports=i;}),null);